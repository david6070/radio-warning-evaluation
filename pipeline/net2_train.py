"""
net2_train.py -- v2 training with a clean outer test split (review item 2).

For each outer fold k (5 link-level folds):
  * the training complement is split BY LINK into fit (70%), early-stopping
    (10%) and calibration (20%) links (seeded, saved in the manifest);
  * the model is fitted on fit links, with epoch / tree count chosen on
    early-stopping links only;
  * the FROZEN model scores the calibration links and the test links of fold k.
net2_eval.py chooses the alarm threshold on fold k's calibration scores and
applies it to fold k's test scores, so nothing about fold k's links is used to
fit the model or to choose its threshold.

Predictors
  rule baselines (no fitting):
    rsrp_now      -RSRP(t)
    rsrp_trend    -(RSRP extrapolated to t+H, least squares over last 10 samples)
    rsrp_ewma     -(exponentially smoothed RSRP, alpha = 0.3)       [smoothed rule]
    joint_margin  -max(RSRP - thr_R, 2 dB * (CQI - thr_C))          [uses RSRP and CQI]
    random        uniform noise                                      [uninformed policy]
  learned:
    logreg_now    logistic regression on current RSRP and CQI only   [same input as joint_margin]
    logreg_rsrp   logistic regression on the 8 RSRP window features  [same input as RSRP rules]
    logreg        logistic regression on all 37 window features
    xgb, lstm, transformer
Usage
  python3 net2_train.py --input <csv> --model xgb --H 30 --seed 0 --out out2 [--rsrp_thr .. --cqi_thr .. --min_run .. --W ..]
"""
import argparse, json, os, time
import numpy as np
import net2_data as N
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = ["rsrp_now", "rsrp_trend", "rsrp_ewma", "joint_margin", "random",
          "logreg_now", "logreg_rsrp", "logreg", "xgb", "lstm", "transformer"]
RULES = {"rsrp_now", "rsrp_trend", "rsrp_ewma", "joint_margin", "random"}


def rule_scores(m, d, H, cfg, seed):
    win, dwin = d["win"], d["dwin"]
    if m == "rsrp_now":
        return -win[:, -1, 0].astype(float)
    if m == "rsrp_trend":
        r = win[:, -10:, 0].astype(float); tt = np.cumsum(dwin[:, -10:], 1); tt = tt - tt[:, -1:]
        tm, rm = tt.mean(1, keepdims=True), r.mean(1, keepdims=True)
        slope = ((tt - tm) * (r - rm)).sum(1) / np.maximum(((tt - tm) ** 2).sum(1), 1e-6)
        return -(r[:, -1] + slope * H)
    if m == "rsrp_ewma":
        e = win[:, 0, 0].astype(float)
        for j in range(1, win.shape[1]):
            e = 0.3 * win[:, j, 0] + 0.7 * e
        return -e
    if m == "joint_margin":
        return -np.maximum(win[:, -1, 0] - cfg.rsrp_thr, 2.0 * (win[:, -1, 3] - cfg.cqi_thr)).astype(float)
    if m == "random":
        return np.random.default_rng(seed + 12345).random(len(win))
    raise ValueError(m)


def tab_features(m, d):
    if m == "logreg_now":
        return d["win"][:, -1, [0, 3]].astype(np.float32)
    F = N.engineered(d)
    return F[:, :8] if m == "logreg_rsrp" else F


def seq_input(d):
    return np.concatenate([d["win"], np.log1p(d["dwin"])[..., None]], axis=2).astype(np.float32)


def split_links(train_links, seed):
    rng = np.random.default_rng(seed)
    u = np.array(sorted(train_links)); rng.shuffle(u)
    n = len(u); n_cal = max(1, int(round(0.2 * n))); n_es = max(1, int(round(0.1 * n)))
    return set(u[n_cal + n_es:]), set(u[n_cal:n_cal + n_es]), set(u[:n_cal])


def fit_torch(kind, Xtr, ytr, Xva, yva, seed, epochs=20, patience=4):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F, Wn = Xtr.shape[2], Xtr.shape[1]

    class LSTM(nn.Module):
        def __init__(s):
            super().__init__(); s.rnn = nn.LSTM(F, 64, 2, batch_first=True, dropout=0.2)
            s.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        def forward(s, x): o, _ = s.rnn(x); return s.head(o[:, -1]).squeeze(-1)

    class TF(nn.Module):
        def __init__(s):
            super().__init__(); s.emb = nn.Linear(F, 64); s.pos = nn.Parameter(torch.zeros(1, Wn, 64))
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, 0.2, batch_first=True), 2)
            s.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(s, x): return s.head(s.enc(s.emb(x) + s.pos).mean(1)).squeeze(-1)

    m = (LSTM() if kind == "lstm" else TF()).to(dev)
    pw = torch.tensor(min(50.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    T = lambda a: torch.tensor(a, device=dev)
    Xt, yt = T(Xtr), T(ytr.astype(np.float32))

    def predict(X):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                out.append(torch.sigmoid(m(T(X[i:i + 4096]))).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    best, state, bad, ep = -1, None, 0, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; opt.zero_grad(); loss = lossf(m(Xt[b]), yt[b]); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        ap = average_precision_score(yva, predict(Xva)) if yva.sum() else 0.0
        if ap > best:
            best, bad, state = ap, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return predict, dict(es_ap=float(best), epochs_run=ep + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--H", type=float, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--out", default="out2")
    ap.add_argument("--rsrp_thr", type=float, default=-100); ap.add_argument("--cqi_thr", type=float, default=7)
    ap.add_argument("--min_run", type=int, default=2); ap.add_argument("--W", type=int, default=20)
    ap.add_argument("--event", default="rc", choices=["rc", "tput_floor", "tput_drop"])
    ap.add_argument("--dl_floor", type=float, default=1000.0); ap.add_argument("--drop_ratio", type=float, default=0.5)
    ap.add_argument("--drop_ref", type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    cfg = N.Cfg(rsrp_thr=a.rsrp_thr, cqi_thr=a.cqi_thr, min_run=a.min_run, W=a.W,
                event=a.event, dl_floor=a.dl_floor, drop_ratio=a.drop_ratio, drop_ref=a.drop_ref)
    df = N.episodes(N.load(a.input, cfg), cfg); d0 = N.build(df, a.H, cfg)
    lab = d0["y"] >= 0
    d = {k: (v[lab] if isinstance(v, np.ndarray) and len(v) == len(lab) else v) for k, v in d0.items()}
    y = d["y"].astype(int); link = d["link"]; fold = N.folds(link, 5, seed=0)
    print(f"{a.model} H={a.H:g} seed={a.seed} cfg={cfg.tag()}: {len(y):,} labelled samples, {100*y.mean():.2f}% positive", flush=True)
    out = dict(rows_all=d["rows"], y_all=d["y"], k=5)
    manifest, info = {}, []
    is_rule = a.model in RULES
    Xall = None if is_rule else (seq_input(d) if a.model in ("lstm", "transformer") else tab_features(a.model, d))
    rule_all = rule_scores(a.model, d, a.H, cfg, a.seed) if is_rule else None
    for k in range(5):
        te = fold == k
        fit_l, es_l, cal_l = split_links(np.unique(link[~te]), seed=1000 * a.seed + k)
        fit = np.isin(link, list(fit_l)); es = np.isin(link, list(es_l)); cal = np.isin(link, list(cal_l))
        manifest[k] = dict(test=sorted(map(str, np.unique(link[te]))), fit=sorted(map(str, fit_l)),
                           es=sorted(map(str, es_l)), cal=sorted(map(str, cal_l)))
        if is_rule:
            score = rule_all; inf = {}
        elif a.model in ("logreg_now", "logreg_rsrp", "logreg"):
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            ft = fit | es                                    # no early stopping needed
            sc = StandardScaler().fit(Xall[ft])
            clf = LogisticRegression(class_weight="balanced", max_iter=3000).fit(sc.transform(Xall[ft]), y[ft])
            score = np.full(len(y), np.nan); sel = cal | te
            score[sel] = clf.predict_proba(sc.transform(Xall[sel]))[:, 1]; inf = {}
        elif a.model == "xgb":
            import xgboost as xgb
            try:
                import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
            spw = (y[fit] == 0).sum() / max(1, (y[fit] == 1).sum())
            clf = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, scale_pos_weight=min(spw, 50), tree_method="hist",
                                    device=dev, eval_metric="aucpr", early_stopping_rounds=40, random_state=a.seed)
            clf.fit(Xall[fit], y[fit], eval_set=[(Xall[es], y[es])], verbose=False)
            score = np.full(len(y), np.nan); sel = cal | te
            score[sel] = clf.predict_proba(Xall[sel])[:, 1]; inf = dict(trees=int(clf.best_iteration) + 1)
        else:
            mu = Xall[fit].reshape(-1, Xall.shape[2]).mean(0); sd = Xall[fit].reshape(-1, Xall.shape[2]).std(0) + 1e-6
            Z = lambda X: ((X - mu) / sd).astype(np.float32)
            predict, inf = fit_torch(a.model, Z(Xall[fit]), y[fit], Z(Xall[es]), y[es], a.seed + 1000 * k, a.epochs)
            score = np.full(len(y), np.nan); sel = cal | te
            score[sel] = predict(Z(Xall[sel]))
        out[f"test_rows_{k}"] = d["rows"][te]; out[f"test_score_{k}"] = score[te]
        out[f"cal_rows_{k}"] = d["rows"][cal]; out[f"cal_score_{k}"] = score[cal]
        inf.update(fold=k, n_fit=int(fit.sum()), n_es=int(es.sum()), n_cal=int(cal.sum()), n_test=int(te.sum()))
        info.append(inf); print(f"  fold {k}: {inf}", flush=True)
    test_all = np.concatenate([out[f"test_score_{k}"] for k in range(5)])
    y_test = np.concatenate([y[fold == k] for k in range(5)])
    print(f"  pooled test AUROC {roc_auc_score(y_test, test_all):.4f}  AUPRC {average_precision_score(y_test, test_all):.4f}"
          f"  base {y.mean():.4f}  [{(time.time()-t0)/60:.1f} min]")
    out["info"] = np.array(json.dumps(dict(model=a.model, H=a.H, seed=a.seed, cfg=cfg.tag(), folds=info, manifest=manifest)))
    np.savez_compressed(os.path.join(a.out, f"s2_{a.model}_{cfg.tag()}_H{a.H:g}_s{a.seed}.npz"), **out)


if __name__ == "__main__":
    main()

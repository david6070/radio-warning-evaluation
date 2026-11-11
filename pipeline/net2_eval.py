"""
net2_eval.py -- action-based evaluation of degradation warnings (v2).

Warning policy semantics
------------------------
* The alarm a_t = 1[s_t >= tau] is evaluated at causally eligible, uncensored
  samples. Ineligible samples (an episode observably under way) and segment
  boundaries switch the alarm off.
* The alarm state holds from one sample until the next. A preventive ACTION
  fires when the alarm switches on, and again at the expiry time (H seconds
  after the previous action) while the alarm is still on.
  Each action claims "an onset will occur in (a, a+H]".
* An action is TRUE if an onset occurs in its window and FALSE if the segment
  is observed through the end of the window without one; otherwise (window
  passes the end of the segment with no onset seen) it is censored and not
  counted. This is the same rule as for sample labels.
* An onset o is DETECTED if some action falls in [t_o - H, t_o); its LEAD is
  t_o minus the earliest such action time (0 < lead <= H).
* FAR = false actions per exposure hour, exposure = summed actual intervals
  of uncensored eligible samples. DUTY = share of exposure time with the
  alarm on.
Sanity: always-off gives DR=0, FAR=0; always-on gives DR~1 with FAR ~ the
number of H-second windows per hour without an onset (it is no longer free);
a random score gives chance-level DR at any FAR.

Thresholds are chosen per outer fold on that fold's CALIBRATION links, scored
by the same frozen model that scores the fold's test links (see net2_train.py).

Usage
  python3 net2_eval.py --input <csv> --H 10 30 60 --scores out2 --B 1000 [--tag r100_c7_m2_w20]
"""
import argparse, glob, os, re, json
import numpy as np, pandas as pd
import net2_data as N

BUDGETS = [0.25, 0.5, 1, 2, 3, 5, 8]
REF = "rsrp_now"


class Track:
    def __init__(s, df, d):
        lab = d["y"] >= 0
        s.H = d["H"]; s.idx = np.flatnonzero(lab)             # positions in the build() arrays
        s.rows, s.t, s.seg, s.y = d["rows"][lab], d["t"][lab], d["seg"][lab], d["y"][lab].astype(int)
        s.link, s.dur = d["link"][lab], d["dur"][lab]
        s.cons = np.r_[False, (np.diff(s.rows) == 1) & (np.diff(s.seg) == 0)]   # contiguous with previous
        seg_end = df.groupby("seg")["t"].max()
        s.seg_end = seg_end.reindex(s.seg).to_numpy()
        o_all = df.loc[df["onset"], ["t", "seg"]]
        s.all_on_key = np.sort(o_all.seg.to_numpy() * 1e9 + (o_all.t.to_numpy() - 1.5e9))
        o = N.onsets_table(df, d); o = o[o.warnable].reset_index(drop=True)
        s.on_t, s.on_seg, s.on_link = o.t.to_numpy(float), o.seg.to_numpy(), o.link_id.to_numpy()
        s.key = s.seg * 1e9 + (s.t - 1.5e9)
        s.on_key = s.on_seg * 1e9 + (s.on_t - 1.5e9)
        s.links = np.unique(s.link)

    def run(s, score, tau, mask=None):
        """Alarm state holds from a sample until the next sample. An action fires
        when the alarm switches on and again every H seconds (at the expiry
        time itself) while the alarm stays on. Returns per-onset detection/lead
        and per-action outcomes; actions whose window passes the end of the
        segment are censored (not counted)."""
        H = s.H
        al = np.nan_to_num(score, nan=-np.inf) >= tau
        if mask is not None:
            al &= mask
        start = al & ~(np.r_[False, al[:-1]] & s.cons)
        st = np.flatnonzero(start)
        last = np.flatnonzero(al & ~np.r_[al[1:] & s.cons[1:], False])     # last sample of each run
        t0 = s.t[st]; t1 = s.t[last] + s.dur[last]                            # alarm on over [t0, t1)
        n = np.maximum(1, np.ceil((t1 - t0) / H - 1e-9)).astype(np.int64)
        run_id = np.repeat(np.arange(len(st)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        ft = t0[run_id] + k * H                                               # action times
        fseg = s.seg[st][run_id]; flink = s.link[st][run_id]
        fkey = fseg * 1e9 + (ft - 1.5e9)
        lo_a = np.searchsorted(s.all_on_key, fkey, side="right")              # onset in (a, a+H] ?
        hi_a = np.searchsorted(s.all_on_key, fkey + H, side="right")
        true = hi_a > lo_a
        # outcome observed: an onset inside the window (true, same rule as a positive
        # label) or the segment observed through a+H (false); otherwise censored
        ok = true | (ft + H <= s.seg_end[st][run_id] + 1e-9)
        fkey_ok, ftrue, flink_ok = fkey[ok], true[ok], flink[ok]
        om = np.isin(s.on_link, np.unique(s.link[mask])) if mask is not None else np.ones(len(s.on_t), bool)
        lo = np.searchsorted(fkey, s.on_key - H, side="left"); hi = np.searchsorted(fkey, s.on_key, side="left")
        det = (hi > lo) & om
        lead = np.full(len(s.on_t), np.nan)
        if len(fkey):
            lead[det] = (s.on_key - fkey[np.clip(lo, 0, len(fkey) - 1)])[det]
        dur = s.dur if mask is None else np.where(mask, s.dur, 0.0)
        exp_h = dur.sum() / 3600
        return dict(det=det, om=om, lead=lead, n_actions=int(ok.sum()), false=~ftrue, fire_link=flink_ok,
                    exp_h=exp_h, duty=(dur * al).sum() / max(dur.sum(), 1e-9))

    def summary(s, r):
        n_on = r["om"].sum()
        return dict(DR=r["det"].sum() / max(n_on, 1), FAR=r["false"].sum() / max(r["exp_h"], 1e-9),
                    duty=r["duty"], actions_per_h=r["n_actions"] / max(r["exp_h"], 1e-9),
                    median_lead=np.nanmedian(r["lead"]) if r["det"].any() else np.nan,
                    lead_ge_10s=(np.nan_to_num(r["lead"], nan=-1) >= 10)[r["om"]].mean() if n_on else np.nan,
                    lead_ge_30s=(np.nan_to_num(r["lead"], nan=-1) >= 30)[r["om"]].mean() if n_on else np.nan,
                    onsets=int(n_on))


def calib_curve(tr, score, mask, n=200, **_):
    """(tau, DR, FAR) on the masked (calibration) links for EVERY candidate
    threshold: +inf (all off) and n quantiles, dense near the top of the score
    distribution. No early stopping: FAR is not monotone in tau (merging alarm
    runs changes when actions renew), so the whole grid is evaluated."""
    v = score[mask]
    q = 1.0 - np.geomspace(1e-5, 0.6, n)
    cands = np.r_[np.inf, np.unique(np.quantile(v, q))[::-1]]
    out = []
    for tau in cands:
        sm = tr.summary(tr.run(score, tau, mask)); out.append((tau, sm["DR"], sm["FAR"]))
    return out


def pick(curve, budget):
    """Max DR subject to FAR <= budget; ties -> larger tau (curve is ordered by decreasing tau)."""
    best_tau, best_dr = np.inf, -1.0
    for tau, dr, far in curve:
        if far <= budget and dr > best_dr + 1e-12:
            best_tau, best_dr = tau, dr
    return best_tau


def per_link(tr, r):
    """Per-link counts for the link bootstrap."""
    L = pd.DataFrame({"link": tr.links})
    on = pd.DataFrame({"link": tr.on_link, "det": r["det"], "om": r["om"]})
    on = on[on.om].groupby("link").agg(n_on=("det", "size"), n_det=("det", "sum"))
    fa = pd.Series(r["fire_link"][r["false"]]).value_counts() if len(r["false"]) else pd.Series(dtype=float)
    ex = pd.Series(tr.dur, index=tr.link).groupby(level=0).sum() / 3600
    L = L.set_index("link").join(on).fillna(0)
    L["n_fa"] = fa.reindex(L.index).fillna(0).to_numpy(); L["exp_h"] = ex.reindex(L.index).fillna(0).to_numpy()
    return L


def load_scores(path, tr, d):
    z = np.load(path, allow_pickle=True)
    lab = d["y"] >= 0
    assert np.array_equal(z["rows_all"], d["rows"][lab]), f"{path}: rows differ from rebuilt task"
    assert np.array_equal(z["y_all"], d["y"][lab]), f"{path}: labels differ from rebuilt task"
    K = int(z["k"])
    pos = {r: i for i, r in enumerate(tr.rows)}
    test = np.full(len(tr.rows), np.nan); fold = np.full(len(tr.rows), -1)
    cal = []
    for k in range(K):
        ti = np.array([pos[r] for r in z[f"test_rows_{k}"]]); test[ti] = z[f"test_score_{k}"]; fold[ti] = k
        ci = np.array([pos[r] for r in z[f"cal_rows_{k}"]]); cs = np.full(len(tr.rows), np.nan); cs[ci] = z[f"cal_score_{k}"]
        cal.append(cs)
    assert (fold >= 0).all(), "some samples have no test score"
    return test, fold, cal, json.loads(str(z["info"]))


def evaluate(tr, test, fold, cal, budgets=BUDGETS):
    curves = []
    for k, cs in enumerate(cal):
        m = ~np.isnan(cs)
        curves.append(calib_curve(tr, np.where(m, cs, -np.inf), m, stop_far=3 * max(budgets)))
    out = []
    for b in budgets:
        tau = np.full(len(test), np.inf); taus = []
        for k in range(len(cal)):
            tk = pick(curves[k], b); taus.append(tk); tau[fold == k] = tk
        r = tr.run(test, tau); sm = tr.summary(r); sm.update(budget=b, taus=taus); out.append((sm, r))
    return out


def perlink_rows(tr, res, H, m, s):
    rows = []
    for sm, r in res:
        L = per_link(tr, r).reset_index().rename(columns={"index": "link"})
        L.insert(0, "budget", sm["budget"]); L.insert(0, "seed", s); L.insert(0, "model", m); L.insert(0, "H", H)
        rows.append(L)
    return rows


def boot(tr, rA, rB, B, rng):
    """Paired link bootstrap of DR and FAR for predictor A and difference A - B."""
    LA, LB = per_link(tr, rA), per_link(tr, rB)
    links = LA.index.to_numpy(); n = len(links)
    a = LA[["n_on", "n_det", "n_fa", "exp_h"]].to_numpy(); bb = LB.loc[links, ["n_on", "n_det", "n_fa", "exp_h"]].to_numpy()
    res = []
    for _ in range(B):
        i = rng.integers(0, n, n); sa = a[i].sum(0); sb = bb[i].sum(0)
        dra, drb = sa[1] / max(sa[0], 1), sb[1] / max(sb[0], 1)
        fa, fb = sa[2] / max(sa[3], 1e-9), sb[2] / max(sb[3], 1e-9)
        res.append((dra, fa, dra - drb, fa - fb))
    q = np.nanpercentile(np.array(res), [2.5, 97.5], axis=0)
    return dict(DR_lo=q[0, 0], DR_hi=q[1, 0], FAR_lo=q[0, 1], FAR_hi=q[1, 1],
                dDR_lo=q[0, 2], dDR_hi=q[1, 2], dFAR_lo=q[0, 3], dFAR_hi=q[1, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--scores", default="out2")
    ap.add_argument("--H", type=float, nargs="+", default=[10, 30, 60]); ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--rsrp_thr", type=float, default=-100); ap.add_argument("--cqi_thr", type=float, default=7)
    ap.add_argument("--min_run", type=int, default=2); ap.add_argument("--W", type=int, default=20)
    ap.add_argument("--event", default="rc", choices=["rc", "tput_floor", "tput_drop"])
    ap.add_argument("--dl_floor", type=float, default=1000.0); ap.add_argument("--drop_ratio", type=float, default=0.5)
    ap.add_argument("--drop_ref", type=int, default=10)
    a = ap.parse_args()
    cfg = N.Cfg(rsrp_thr=a.rsrp_thr, cqi_thr=a.cqi_thr, min_run=a.min_run, W=a.W,
                event=a.event, dl_floor=a.dl_floor, drop_ratio=a.drop_ratio, drop_ref=a.drop_ref)
    rng = np.random.default_rng(20261019)
    df = N.episodes(N.load(a.input, cfg), cfg)
    info, acc_H = N.summary(df, a.H, cfg)
    json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()},
              open(f"{a.scores}/eval2_dataset_{cfg.tag()}.json", "w"), indent=1, default=str)
    acc_H.to_csv(f"{a.scores}/eval2_accounting_{cfg.tag()}.csv", index=False)
    print(info); print(acc_H.round(3).to_string(index=False))
    from sklearn.metrics import roc_auc_score, average_precision_score
    rows_ops, rows_acc, rows_pl = [], [], []
    for H in a.H:
        d = N.build(df, H, cfg); tr = Track(df, d)
        files = sorted(glob.glob(f"{a.scores}/s2_*_{cfg.tag()}_H{H:g}_s*.npz"))
        res = {}
        for f in files:
            m, s = re.search(r"s2_(.+)_" + re.escape(cfg.tag()) + r"_H[\d.]+_s(\d+)\.npz", f).groups(); s = int(s)
            test, fold, cal, inf = load_scores(f, tr, d)
            rows_acc.append(dict(H=H, model=m, seed=s, auroc=roc_auc_score(tr.y, test),
                                 auprc=average_precision_score(tr.y, test), base_rate=tr.y.mean()))
            res[(m, s)] = evaluate(tr, test, fold, cal)
            rows_pl += perlink_rows(tr, res[(m, s)], H, m, s)
            print(f"H={H:g} {m} s{s}: " + " ".join(f"[{x[0]['budget']}: DR {x[0]['DR']:.2f} FAR {x[0]['FAR']:.2f}]" for x in res[(m, s)]), flush=True)
        ref = res.get((REF, 0))
        for (m, s), lst in res.items():
            for bi, (sm, r) in enumerate(lst):
                row = dict(H=H, model=m, seed=s, **{k: v for k, v in sm.items() if k != "taus"})
                if s == 0 and ref is not None and a.B > 0:
                    row.update(boot(tr, r, ref[bi][1], a.B, rng))
                rows_ops.append(row)
    pd.DataFrame(rows_acc).to_csv(f"{a.scores}/eval2_accuracy_{cfg.tag()}.csv", index=False)
    pd.DataFrame(rows_ops).to_csv(f"{a.scores}/eval2_ops_{cfg.tag()}.csv", index=False)
    if rows_pl:
        pd.concat(rows_pl).to_csv(f"{a.scores}/eval2_perlink_{cfg.tag()}.csv.gz", index=False)
    print("wrote", f"{a.scores}/eval2_*_{cfg.tag()}.*")


if __name__ == "__main__":
    main()

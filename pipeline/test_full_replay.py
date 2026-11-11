"""
test_full_replay.py -- checks that the evaluator's results equal a full causal replay.

net2_train.py exports scores only for eligible moments whose label is observed
(y >= 0), and net2_eval.Track builds alarm runs from those moments. A deployed
controller would also score the censored moments at the end of each segment.
This test replays the policy over ALL eligible moments (censored included),
fires and renews actions, and only afterwards classifies each action as
true / false / censored. It then compares DR, FAR, counted actions, duty and
leads with Track.run on the labelled-only subset, for many scores and thresholds.

Why they must agree: a moment is censored iff no onset lies in (t, t+H] and the
segment ends before t+H. Every later eligible moment in that segment is then
censored too and has no onset ahead of it, so any action fired or renewed in
that tail has no onset in its window and a window passing the segment end:
it is censored whichever way the tail is handled.

  python3 test_full_replay.py [--input <processed csv>]   (synthetic data if omitted)
"""
import argparse, os, tempfile
import numpy as np, pandas as pd
import net2_data as N, net2_eval as E


def synthetic_csv(path, n_links=12, seed=0):
    rng = np.random.default_rng(seed); parts = []
    for L in range(n_links):
        n = int(rng.integers(400, 1500)); dt = rng.choice([1, 1, 1, 2, 3], n).astype(float)
        dt[rng.random(n) < 0.004] = 45.0                                   # gaps > 30 s split segments
        ts = pd.Timestamp("2021-01-01") + pd.to_timedelta(np.cumsum(dt), unit="s")
        rsrp = -95 + np.cumsum(rng.normal(0, 1.2, n)).clip(-25, 25) + rng.normal(0, 2, n)
        cqi = np.clip(np.round(9 + (rsrp + 95) / 4 + rng.normal(0, 2, n)), 0, 15)
        df = pd.DataFrame(dict(link_id=f"L{L}", timestamp=ts, RSRP=rsrp, RSRQ=rng.normal(-11, 2, n),
                               SINR=rng.normal(5, 4, n), CQI=cqi))
        df.loc[rng.random(n) < 0.01, "CQI"] = np.nan                       # missing CQI also splits segments
        df.loc[rng.random(n) < 0.05, "SINR"] = np.nan
        parts.append(df)
    pd.concat(parts).to_csv(path, index=False)


def full_replay(df, d, score_all, tau):
    """Independent loop implementation over all eligible moments."""
    H = d["H"]; t, seg, y, dur, rows = d["t"], d["seg"], d["y"], d["dur"], d["rows"]
    seg_end = df.groupby("seg")["t"].max().to_dict()
    on_by_seg = {g: np.sort(v.to_numpy(float)) for g, v in df.loc[df["onset"]].groupby("seg")["t"]}
    al = np.nan_to_num(score_all, nan=-np.inf) >= tau
    n = len(t); actions = []                                              # (time, seg)
    i = 0
    while i < n:
        if not al[i]:
            i += 1; continue
        j = i                                                             # extend run over contiguous alarmed rows
        while j + 1 < n and al[j + 1] and rows[j + 1] == rows[j] + 1 and seg[j + 1] == seg[j]:
            j += 1
        t0, t1 = t[i], t[j] + dur[j]
        a = t0
        while True:
            actions.append((a, seg[i])); a += H
            if a >= t1 - 1e-9:
                break
        i = j + 1
    n_true = n_false = 0; times_by_seg = {}
    for a, g in actions:
        on = on_by_seg.get(g, np.array([]))
        tr = bool(((on > a) & (on <= a + H + 1e-9)).any())
        if tr: n_true += 1
        elif a + H <= seg_end[g] + 1e-9: n_false += 1
        times_by_seg.setdefault(g, []).append(a)
    o = N.onsets_table(df, d); o = o[o.warnable]
    det, lead = [], []
    for to, g in zip(o.t, o.seg):
        ac = np.array(times_by_seg.get(g, []))
        hit = ac[(ac >= to - H - 1e-9) & (ac < to - 1e-9)] if len(ac) else ac
        det.append(len(hit) > 0); lead.append(to - hit.min() if len(hit) else np.nan)
    unc = y >= 0; exp_h = dur[unc].sum() / 3600
    return dict(DR=np.mean(det) if det else 0.0, FAR=n_false / exp_h, n_counted=n_true + n_false,
                duty=(dur * al)[unc].sum() / dur[unc].sum(), leads=np.array(lead))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input"); ap.add_argument("--n_scores", type=int, default=6)
    a = ap.parse_args()
    path = a.input
    if path is None:
        path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name; synthetic_csv(path)
    cfg = N.Cfg(); df = N.episodes(N.load(path, cfg), cfg)
    rng = np.random.default_rng(7); ok = True; n_cmp = 0
    for H in (10.0, 30.0, 60.0):
        d = N.build(df, H, cfg); lab = d["y"] >= 0; tr = E.Track(df, d)
        print(f"H={H:g}: {len(d['y'])} eligible, {(~lab).sum()} censored, {int(tr.on_t.size)} warnable onsets")
        scores = [rng.random(len(lab)), -d["win"][:, -1, 0],                           # random, RSRP level
                  (d["next_onset"] - d["t"] <= H).astype(float) + 0.1 * rng.random(len(lab))]  # near-oracle
        scores += [rng.random(len(lab)) ** rng.uniform(0.2, 5) for _ in range(a.n_scores - 3)]
        for s_all in scores:
            for q in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
                tau = np.quantile(s_all[lab], q)
                ref = full_replay(df, d, s_all, tau)
                r = tr.run(s_all[lab], tau); sm = tr.summary(r)
                same = (np.isclose(ref["DR"], sm["DR"]) and np.isclose(ref["FAR"], sm["FAR"])
                        and ref["n_counted"] == sm.get("actions_per_h", 0) * 0 + r["n_actions"]
                        and np.isclose(ref["duty"], sm["duty"])
                        and np.allclose(ref["leads"], r["lead"], equal_nan=True))
                n_cmp += 1
                if not same:
                    ok = False; print("  MISMATCH", H, q, ref["DR"], sm["DR"], ref["FAR"], sm["FAR"], ref["n_counted"], r["n_actions"])
    print(("PASS" if ok else "FAIL") + f": full causal replay equals labelled-only evaluation in {n_cmp} comparisons")
    if a.input is None: os.unlink(path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

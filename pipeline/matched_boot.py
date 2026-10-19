"""
matched_boot.py -- paired link-bootstrap uncertainty for detection at a MATCHED
realised false-action rate (review item 2). Uses eval2_perlink_*.csv.gz.

For each resample of links (same links for every predictor): recompute DR and FAR
of every predictor/seed at each of its calibrated operating points, form the upper
DR envelope over FAR, interpolate in log(FAR) at the target rate, average over the
seeds that span the rate, and take the difference from RSRP level.
This is an empirical frontier (an operator choosing among the calibrated operating
points), not a single frozen policy at exactly that rate.
  python3 matched_boot.py --perlink out2_cork5g/eval2_perlink_r100_c7_m2_w20.csv.gz --out out2_cork5g --rates 1 2 4 --B 2000
"""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--perlink", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rates", type=float, nargs="+", default=[1, 2, 4]); ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--ref", default="rsrp_now"); a = ap.parse_args()
P = pd.read_csv(a.perlink)
rng = np.random.default_rng(20261019)


def env_interp(far, dr, r):
    """far, dr: (..., n_budgets). Upper envelope of DR over FAR, interpolated in log FAR."""
    o = np.argsort(far, axis=-1)
    f = np.take_along_axis(far, o, -1); d = np.maximum.accumulate(np.take_along_axis(dr, o, -1), axis=-1)
    out = np.full(f.shape[:-1], np.nan)
    for idx in np.ndindex(f.shape[:-1]):
        ff, dd = f[idx], d[idx]; ok = ff > 0
        if ok.sum() == 0: continue
        x = np.log(ff[ok]); y = dd[ok]; l = np.log(r)
        if x.min() <= l <= x.max(): out[idx] = np.interp(l, x, y)
    return out


rows = []
for H, g in P.groupby("H"):
    links = np.array(sorted(g.link.unique())); L = len(links); li = {l: i for i, l in enumerate(links)}
    budgets = sorted(g.budget.unique())
    arr = {}                                                     # (model, seed) -> (n_budget, L, 4)
    for (m, s), q in g.groupby(["model", "seed"]):
        A = np.zeros((len(budgets), L, 4))
        for bi, b in enumerate(budgets):
            qq = q[q.budget == b]
            A[bi, [li[x] for x in qq.link], :] = qq[["n_on", "n_det", "n_fa", "exp_h"]].to_numpy()
        arr[(m, s)] = A
    models = sorted({m for m, _ in arr})
    idx = rng.integers(0, L, (a.B, L))

    def stats(A, ii):
        S = A[:, ii, :].sum(axis=-2) if ii.ndim == 2 else A[:, ii, :].sum(axis=1)
        return S

    est, boot = {}, {}
    for m in models:
        seeds = sorted(s for mm, s in arr if mm == m)
        for r in a.rates:
            vals, bvals = [], []
            for s in seeds:
                A = arr[(m, s)]
                S0 = A.sum(axis=1); dr0 = S0[:, 1] / np.maximum(S0[:, 0], 1); far0 = S0[:, 2] / np.maximum(S0[:, 3], 1e-9)
                vals.append(env_interp(far0[None, :], dr0[None, :], r)[0])
                Sb = A[:, idx, :].sum(axis=2)                    # (n_budget, B, 4)
                drb = (Sb[..., 1] / np.maximum(Sb[..., 0], 1)).T; farb = (Sb[..., 2] / np.maximum(Sb[..., 3], 1e-9)).T
                bvals.append(env_interp(farb, drb, r))
            est[(m, r)] = (np.nanmean(vals) if np.isfinite(vals).any() else np.nan, int(np.isfinite(vals).sum()), len(seeds))
            with np.errstate(all="ignore"):
                boot[(m, r)] = np.nanmean(np.vstack(bvals), axis=0)
    for m in models:
        for r in a.rates:
            e, n_ok, n_s = est[(m, r)]; b = boot[(m, r)]; ref = boot[(a.ref, r)]
            d = b - ref; good = np.isfinite(d)
            rows.append(dict(H=H, model=m, rate=r, DR=e, seeds_spanning=n_ok, seeds=n_s,
                             DR_lo=np.nanpercentile(b, 2.5) if np.isfinite(b).any() else np.nan,
                             DR_hi=np.nanpercentile(b, 97.5) if np.isfinite(b).any() else np.nan,
                             dDR=e - est[(a.ref, r)][0],
                             dDR_lo=np.percentile(d[good], 2.5) if good.sum() > 50 else np.nan,
                             dDR_hi=np.percentile(d[good], 97.5) if good.sum() > 50 else np.nan,
                             boot_support=float(good.mean())))
R = pd.DataFrame(rows); tag = a.perlink.split("eval2_perlink_")[1].replace(".csv.gz", "")
R.to_csv(f"{a.out}/eval2_matched_boot_{tag}.csv", index=False)
pd.set_option("display.width", 200); print(R[R.rate == 2].round(3).to_string(index=False))

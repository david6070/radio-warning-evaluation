"""
proxy_check.py -- does the degradation proxy (RSRP < thr and CQI < thr for >= m
samples) coincide with a drop in delivered throughput? Cork traces only
(DL_bitrate, kbit/s). Restricted to sessions where the UE was actively
downloading or streaming (DL_bitrate available).
  python3 proxy_check.py --input cork/cork5g_v2.csv --out out2_cork5g
"""
import argparse, json, numpy as np, pandas as pd
import net2_data as N
ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); cfg = N.Cfg()
df = N.episodes(N.load(a.input, cfg), cfg)
df["DL"] = pd.to_numeric(df["DL_bitrate"], errors="coerce")
res = {}
for app, g in df.groupby("app"):
    g = g[g.DL.notna()]
    res[app] = dict(samples=len(g), median_DL_in_episode=float(g.loc[g.in_ep, "DL"].median()),
                    median_DL_outside=float(g.loc[~g.in_ep, "DL"].median()),
                    share_DL_below_1Mbps_in_episode=float((g.loc[g.in_ep, "DL"] < 1000).mean()),
                    share_DL_below_1Mbps_outside=float((g.loc[~g.in_ep, "DL"] < 1000).mean()))
# around onsets: median DL in the 10 s before vs the 10 s after (same segment)
t = df.t.to_numpy(); seg = df.seg.to_numpy(); DL = df.DL.to_numpy(); app = df.app.to_numpy()
rows = []
for i in np.flatnonzero(df.onset.to_numpy()):
    s = seg == seg[i]
    pre = s & (t >= t[i] - 10) & (t < t[i]); post = s & (t >= t[i]) & (t < t[i] + 10)
    if np.isfinite(DL[pre]).any() and np.isfinite(DL[post]).any():
        rows.append(dict(app=app[i], pre=np.nanmedian(DL[pre]), post=np.nanmedian(DL[post])))
R = pd.DataFrame(rows)
if len(R):
    R["ratio"] = (R.post + 1) / (R.pre + 1)
    res["onsets"] = R.groupby("app").agg(n=("ratio", "size"), median_pre=("pre", "median"), median_post=("post", "median"),
                                        median_ratio=("ratio", "median"), share_drop_50pct=("ratio", lambda r: float((r < 0.5).mean()))).to_dict("index")
json.dump(res, open(f"{a.out}/proxy_check.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))

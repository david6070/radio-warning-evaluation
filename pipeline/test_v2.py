"""
test_v2.py -- checks that must pass before any training (run on Juno too).
  python3 test_v2.py --input <processed csv>
1. no future information in features (perturb the future, past windows unchanged)
2. causal eligibility (perturb t+1, eligibility of t unchanged)
3. evaluator sanity: always-off, always-on, oracle, random
4. the review's counter-example: a constant always-on score is no longer free
"""
import argparse, numpy as np, pandas as pd, tempfile, os
import net2_data as N, net2_eval as E

ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--H", type=float, default=30)
a = ap.parse_args()
cfg = N.Cfg(); ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"  ({extra})" if extra else "")); ok &= bool(cond)

raw = pd.read_csv(a.input, low_memory=False)
df = N.episodes(N.load(a.input, cfg), cfg); d = N.build(df, a.H, cfg)

# 1. future perturbation: change all values after the midpoint of every link; features of samples before it must not change
raw2 = raw.copy(); raw2["timestamp"] = pd.to_datetime(raw2["timestamp"])
cut = raw2.groupby("link_id")["timestamp"].transform(lambda s: s.iloc[len(s) // 2])
fut = raw2["timestamp"] > cut
rng = np.random.default_rng(0)
for c in ["RSRP", "RSRQ", "SINR", "CQI"]:
    v = pd.to_numeric(raw2[c], errors="coerce").astype(float); raw2[c] = v
    raw2.loc[fut, c] = v[fut] + rng.normal(0, 5, fut.sum())
raw2.loc[fut & (rng.random(len(raw2)) < 0.2), ["RSRQ", "SINR"]] = np.nan     # also inject missingness in the future
tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name; raw2.to_csv(tmp, index=False)
df2 = N.episodes(N.load(tmp, cfg), cfg)
cutt = (cut - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
cut_by_link = dict(zip(raw2["link_id"], cutt))
past = df["t"].to_numpy() <= np.array([cut_by_link[l] for l in df["link_id"]])
m = df[past][["link_id", "t"]].merge(df2[["link_id", "t", "RSRP", "RSRQ", "SINR", "CQI", "RSRQ_miss", "SINR_miss"]], on=["link_id", "t"], how="left")
same = np.allclose(df[past][["RSRP", "RSRQ", "SINR", "CQI", "RSRQ_miss", "SINR_miss"]].to_numpy(),
                   m[["RSRP", "RSRQ", "SINR", "CQI", "RSRQ_miss", "SINR_miss"]].to_numpy(), equal_nan=True)
check("past measurements/imputations unaffected by future changes", same)
conf_same = np.array_equal(df[past]["conf_ep"].to_numpy(),
                           df[past][["link_id", "t"]].merge(df2[["link_id", "t", "conf_ep"]], on=["link_id", "t"])["conf_ep"].to_numpy())
check("causal eligibility (conf_ep) of past samples unaffected by future changes", conf_same)
os.unlink(tmp)

# 2. eligibility does not look ahead: conf_ep[t] uses degraded[t-m+1..t] only
deg = df["degraded"].to_numpy(); seg = df["seg"].to_numpy(); conf = df["conf_ep"].to_numpy()
recomp = np.zeros(len(df), bool)
for i in range(len(df)):
    j0 = i - cfg.min_run + 1
    recomp[i] = j0 >= 0 and seg[j0] == seg[i] and deg[j0:i + 1].all()
check("conf_ep equals 'last min_run samples degraded' exactly", np.array_equal(recomp, conf))

# 3. evaluator sanity
tr = E.Track(df, d); n = len(tr.y); H = a.H
off = tr.summary(tr.run(np.zeros(n), np.inf))
check("always-off: DR = 0 and FAR = 0", off["DR"] == 0 and off["FAR"] == 0, off)
on = tr.summary(tr.run(np.ones(n), 0.5))
check("always-on: DR close to 1", on["DR"] > 0.95, f"DR {on['DR']:.3f}")
check("always-on: FAR is large (not free)", on["FAR"] > 0.3 * 3600 / H * (1 - tr.y.mean()), f"FAR {on['FAR']:.1f}/h, duty {on['duty']:.2f}")
check("always-on: every lead <= H", np.nanmax(tr.run(np.ones(n), 0.5)["lead"]) <= H + 1e-9)
check("always-on: DR exactly 1 (warning held between samples)", on["DR"] > 0.999, f"DR {on['DR']:.4f}")
orc = tr.summary(tr.run(tr.y.astype(float), 0.5))
check("oracle (score = label): FAR = 0 and DR close to 1", orc["FAR"] == 0 and orc["DR"] > 0.95, orc)
rnd = np.random.default_rng(1).random(n)
for q in (0.99, 0.95):
    rr = tr.summary(tr.run(rnd, np.quantile(rnd, q)))
    print(f"     random score, top {100*(1-q):.0f}% alarmed: DR {rr['DR']:.3f}, FAR {rr['FAR']:.2f}/h, duty {rr['duty']:.3f}")

# 4. review counter-example: one link, constant score, onset near the end -> must cost false alarms
one = tr.link == tr.link[np.argmax(np.bincount(pd.factorize(tr.link)[0]))]
r1 = tr.summary(tr.run(np.where(one, 1.0, 0.0), 0.5, mask=one))
check("constant score on a long link is charged for false actions", r1["FAR"] > 0, r1)

# 5. boundary case from the review: measurements t = 0..45 s, W = 20, H = 30, a 2-sample
#    degradation starting at t = 44 (end of segment). The onset is warnable; always-on and
#    oracle must detect it, and the true action must be counted (not censored).
import tempfile as _tf
tt = pd.date_range("2020-01-01", periods=46, freq="s")
b_df = pd.DataFrame({"link_id": "edge", "timestamp": tt, "RSRP": -80.0, "RSRQ": -10.0, "SINR": 10.0, "CQI": 12.0})
b_df.loc[44:, ["RSRP", "CQI"]] = [-110.0, 3.0]
bp = _tf.NamedTemporaryFile(suffix=".csv", delete=False).name; b_df.to_csv(bp, index=False)
bdf = N.episodes(N.load(bp, cfg), cfg); bd = N.build(bdf, 30, cfg); btr = E.Track(bdf, bd)
bon = btr.run(np.ones(len(btr.y)), 0.5); bor = btr.run(btr.y.astype(float), 0.5)
check("boundary: onset at segment end is warnable", len(btr.on_t) == 1, f"warnable {len(btr.on_t)}")
check("boundary: always-on detects it and its true action is counted",
      bon["det"].all() and (~bon["false"]).sum() >= 1, f"det {bon['det']}, true actions {(~bon['false']).sum()}")
check("boundary: oracle detects it with no false or censored-true actions",
      bor["det"].all() and bor["false"].sum() == 0 and bor["n_actions"] >= 1, f"actions {bor['n_actions']}, false {bor['false'].sum()}")
os.unlink(bp)

# 6. threshold search evaluates the complete candidate grid (FAR is not monotone in tau)
m_all = np.ones(len(tr.y), bool)
cv = E.calib_curve(tr, rnd, m_all, n=50)
check("calibration search evaluates every candidate threshold", len(cv) == 1 + len(np.unique(np.quantile(rnd, 1.0 - np.geomspace(1e-5, 0.6, 50)))), f"{len(cv)} candidates")
far = np.array([c[2] for c in cv]); print(f"     FAR non-monotone in tau on this data: {bool((np.diff(far) < -1e-9).any())}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")

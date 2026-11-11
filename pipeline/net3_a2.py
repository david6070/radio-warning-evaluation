"""
net3_a2.py -- A2-like serving-cell rule with time-to-trigger and hysteresis, evaluated
with exactly the protocol of net2_train.py / net2_eval.py (same outer folds, same
fit/early-stop/calibration split for seed 0, threshold chosen on calibration links).

Rule (3GPP TS 38.331 event A2 style, applied to the serving-cell RSRP):
  enter  when RSRP + hys < theta for the last k measurements (time-to-trigger k),
  leave  when RSRP - hys > theta;
  the state resets at segment boundaries and ineligible moments, as for every predictor.
The threshold theta plays the role of tau: it is calibrated per outer fold on the
calibration links for every budget, over a grid of candidate values (plus "off").
Each alarm sequence is evaluated by net2_eval.Track, so actions, renewals, censoring,
DR, FAR, duty and lead are computed exactly as for the other predictors.

Outputs (same formats as net2_eval.py, model names a2_k{k}_h{hys}):
  <out>/eval3_a2_ops_<tag>.csv and <out>/eval3_a2_perlink_<tag>.csv.gz
The per-link file can be passed to matched_boot.py together with the main per-link file.
  python3 net3_a2.py --input ../cork5g_v2.csv --out ../out2_cork5g --H 10 30 60 --variants 3:2 1:2 5:2 3:0
"""
import argparse
import numpy as np, pandas as pd
import net2_data as N, net2_eval as E
from net2_train import split_links


def a2_alarm(rsrp_win, cons, theta, k, hys):
    """rsrp_win: (n, W) RSRP window ending at each labelled moment (Track order);
    cons: contiguity with the previous labelled moment. Vectorised state machine:
    the alarm is on at i iff the latest 'decisive' moment j <= i in the current
    contiguous block is an entering moment, where decisive = entering condition met
    or leaving condition met (a block start without entering counts as leaving)."""
    enter = (rsrp_win[:, -k:] + hys < theta).all(axis=1)
    leave = (rsrp_win[:, -1] - hys > theta) | ~np.asarray(cons, bool)
    decisive = enter | leave
    n = len(enter); idx = np.where(decisive, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, enter[np.maximum(last, 0)], False)


def a2_alarm_loop(rsrp_win, cons, theta, k, hys):
    """Reference implementation (used by the self-test)."""
    enter = (rsrp_win[:, -k:] + hys < theta).all(axis=1)
    stay_ok = rsrp_win[:, -1] - hys <= theta
    al = np.zeros(len(enter), bool); prev = False
    for i in range(len(enter)):
        if not cons[i]:
            prev = False
        prev = bool(enter[i] or (prev and stay_ok[i])); al[i] = prev
    return al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--H", type=float, nargs="+", default=[10, 30, 60])
    ap.add_argument("--variants", nargs="+", default=["3:2", "1:2", "5:2", "3:0"], help="k:hys (samples:dB)")
    ap.add_argument("--n_theta", type=int, default=120)
    ap.add_argument("--rsrp_thr", type=float, default=-100); ap.add_argument("--cqi_thr", type=float, default=7)
    ap.add_argument("--min_run", type=int, default=2); ap.add_argument("--W", type=int, default=20)
    ap.add_argument("--event", default="rc", choices=["rc", "tput_floor", "tput_drop"])
    ap.add_argument("--dl_floor", type=float, default=1000.0); ap.add_argument("--drop_ratio", type=float, default=0.5)
    ap.add_argument("--drop_ref", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        rng = np.random.default_rng(0); n = 20000
        win = -100 + np.cumsum(rng.normal(0, 1.5, (n, 1)), 0) + rng.normal(0, 2, (n, 20))
        cons = rng.random(n) > 0.02
        for th in (-110, -100, -95):
            for k, h in ((1, 0), (3, 2), (5, 2)):
                assert np.array_equal(a2_alarm(win, cons, th, k, h), a2_alarm_loop(win, cons, th, k, h)), (th, k, h)
        print("PASS a2_alarm vectorised == loop"); return
    cfg = N.Cfg(rsrp_thr=a.rsrp_thr, cqi_thr=a.cqi_thr, min_run=a.min_run, W=a.W,
                event=a.event, dl_floor=a.dl_floor, drop_ratio=a.drop_ratio, drop_ref=a.drop_ref)
    df = N.episodes(N.load(a.input, cfg), cfg)
    ops, pls = [], []
    for H in a.H:
        d = N.build(df, H, cfg); tr = E.Track(df, d)
        lab = d["y"] >= 0
        rsrp = d["win"][lab][:, :, 0].astype(float)              # same order as Track (labelled rows)
        link = d["link"][lab]; fold = N.folds(link, 5, seed=0)
        thetas = np.unique(np.quantile(rsrp[:, -1], np.linspace(0.0005, 0.6, a.n_theta)))
        for v in a.variants:
            k, hys = int(v.split(":")[0]), float(v.split(":")[1]); name = f"a2_k{k}_h{hys:g}"
            cache = {th: a2_alarm(rsrp, tr.cons, th, k, hys) for th in thetas}
            cand = [None] + list(thetas)                        # None = alarm off; then more and more alarms
            tau_test = {b: np.zeros(len(link), bool) for b in E.BUDGETS}
            for kf in range(5):
                te = fold == kf
                _, _, cal_l = split_links(np.unique(link[~te]), seed=kf)      # seed 0: 1000*0 + k
                cal = np.isin(link, list(cal_l))
                curve = []
                for th in cand:
                    al = np.zeros(len(link), bool) if th is None else cache[th]
                    sm = tr.summary(tr.run(al.astype(float), 0.5, cal)); curve.append((th, sm["DR"], sm["FAR"]))
                for b in E.BUDGETS:
                    best, best_dr = None, -1.0
                    for th, dr, far in curve:
                        if far <= b and dr > best_dr + 1e-12:
                            best, best_dr = th, dr
                    if best is not None:
                        tau_test[b][te] = cache[best][te]
            res = []
            for b in E.BUDGETS:
                r = tr.run(tau_test[b].astype(float), 0.5); sm = tr.summary(r); sm["budget"] = b
                ops.append(dict(H=H, model=name, seed=0, **sm)); res.append((sm, r))
            pls += E.perlink_rows(tr, res, H, name, 0)
            print(f"H={H:g} {name}: " + " ".join(f"[{x[0]['budget']}: DR {x[0]['DR']:.2f} FAR {x[0]['FAR']:.2f}]" for x in res), flush=True)
    pd.DataFrame(ops).to_csv(f"{a.out}/eval3_a2_ops_{cfg.tag()}.csv", index=False)
    pd.concat(pls).to_csv(f"{a.out}/eval3_a2_perlink_{cfg.tag()}.csv.gz", index=False)
    print("wrote", f"{a.out}/eval3_a2_*_{cfg.tag()}.*")


if __name__ == "__main__":
    main()

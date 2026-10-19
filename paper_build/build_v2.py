"""Builds every table (LaTeX) and figure of the v2 manuscript from the Juno eval2 outputs in data/out2_*."""
import json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.edgecolor": "0.35", "xtick.color": "0.35", "ytick.color": "0.35", "grid.color": "0.9"})
DS = [("cork5g", "Cork 5G"), ("cork4g", "Cork 4G"), ("manaus", "Manaus (exploratory)")]
MAIN = "r100_c7_m2_w20"
ORDER = ["random", "rsrp_now", "rsrp_trend", "rsrp_ewma", "joint_margin", "logreg_now", "logreg_rsrp", "logreg", "xgb", "lstm", "transformer"]
NAME = {"random": "Random (uninformed)", "rsrp_now": "RSRP level", "rsrp_trend": "RSRP trend", "rsrp_ewma": "RSRP smoothed",
        "joint_margin": "RSRP+CQI margin", "logreg_now": "LR, current RSRP+CQI", "logreg_rsrp": "LR, RSRP window",
        "logreg": "LR, all features", "xgb": "XGBoost", "lstm": "LSTM", "transformer": "Transformer"}
SHOW = ["rsrp_now", "joint_margin", "logreg", "xgb", "lstm", "transformer", "random"]
COL = {"rsrp_now": "#1a1a1a", "joint_margin": "#8c8c8c", "logreg": "#56B4E9", "xgb": "#E69F00", "lstm": "#0072B2",
       "transformer": "#009E73", "random": "#c8c8c8"}
LS = {"rsrp_now": "-", "joint_margin": (0, (4, 2)), "random": (0, (1, 1.5))}
MK = {"rsrp_now": "s", "joint_margin": "D", "logreg": "o", "xgb": "^", "lstm": "v", "transformer": "h", "random": "x"}
ops = {d: pd.read_csv(f"data/out2_{d}/eval2_ops_{MAIN}.csv") for d, _ in DS}
acc = {d: pd.read_csv(f"data/out2_{d}/eval2_accuracy_{MAIN}.csv") for d, _ in DS}
acct = {d: pd.read_csv(f"data/out2_{d}/eval2_accounting_{MAIN}.csv") for d, _ in DS}
info = {d: json.load(open(f"data/out2_{d}/eval2_dataset_{MAIN}.json")) for d, _ in DS}
N = {}

def interp(g, fa):
    g = g[g.FAR > 0].sort_values("FAR"); x = np.log(g.FAR.values); y = np.maximum.accumulate(g.DR.values); l = np.log(fa)
    return np.nan if len(x) == 0 or l < x.min() or l > x.max() else float(np.interp(l, x, y))

match = []
for d, _ in DS:
    for (H, m, s), g in ops[d].groupby(["H", "model", "seed"]):
        match.append(dict(D=d, H=H, model=m, seed=s, d1=interp(g, 1), d2=interp(g, 2), d4=interp(g, 4)))
M = pd.DataFrame(match); M.to_csv("tables/matched_by_seed.csv", index=False)

def fmt(v, nd=2):
    return "--" if pd.isna(v) else f"{v:.{nd}f}"

# ---- Table 1: datasets
rows = []
for d, lab in DS:
    i, a = info[d], acct[d].set_index("H")
    rows.append([lab, f"{i['rows_used']:,}", f"{i['links']:,} ({i['links_with_onset']})", f"{i['observed_h']:.1f}",
                 f"{a.loc[30,'exposure_h']:.1f}", f"{i['onsets']:,} ({a.loc[30,'warnable_onsets']:,})",
                 f"{100*i['degraded_share']:.1f}", " / ".join(f"{100*a.loc[h,'positive_rate']:.1f}" for h in (10, 30, 60))])
    N[d] = dict(info=i, acct=a.reset_index().to_dict("records"))
T1 = "\n".join(" & ".join(r) + r" \\" for r in rows)
open("tables/t1_data.tex", "w").write(T1)

# ---- Table 2: AUPRC (mean over seeds)
A = pd.concat([acc[d].assign(D=d) for d, _ in DS]).groupby(["D", "H", "model"]).auprc.mean().reset_index()
cols = [(d, h) for d, _ in DS for h in (10, 30, 60)]
lines = [f"Base rate & " + " & ".join(fmt(acc[d][acc[d].H == h].base_rate.iloc[0], 3) for d, h in cols) + r" \\ \midrule"]
best = {c: A[(A.D == c[0]) & (A.H == c[1]) & ~A.model.eq("random")].auprc.max() for c in cols}
for m in ORDER:
    vals = []
    for c in cols:
        v = A[(A.D == c[0]) & (A.H == c[1]) & (A.model == m)].auprc.iloc[0]
        vals.append((r"\textbf{" + fmt(v, 3) + "}") if abs(v - best[c]) < 1e-12 else fmt(v, 3))
    lines.append(NAME[m] + " & " + " & ".join(vals) + r" \\")
open("tables/t2_auprc.tex", "w").write("\n".join(lines))

# ---- Table 3: matched detection at 2/h (mean over seeds, [min,max] for 3-seed models)
lines = []
G = M.groupby(["D", "H", "model"]).d2
mean, cnt = G.mean(), G.count()
for m in ORDER:
    vals = []
    for c in cols:
        key = (c[0], float(c[1]), m)
        v = mean.get(key, np.nan); n = cnt.get(key, 0)
        colvals = mean[[k for k in mean.index if k[0] == c[0] and k[1] == float(c[1]) and k[2] != "random"]]
        s = fmt(v)
        if not pd.isna(v) and abs(v - colvals.max()) < 1e-12: s = r"\textbf{" + s + "}"
        if m in ("xgb", "lstm", "transformer") and 0 < n < 3: s += r"$^{" + str(n) + "}$"
        vals.append(s)
    lines.append(NAME[m] + " & " + " & ".join(vals) + r" \\")
open("tables/t3_matched2.tex", "w").write("\n".join(lines))

# ---- Table 4: paired bootstrap vs RSRP level, seed 0, nominal 2/h, H = 30 and 60
lines = []
for d, lab in DS:
    o = ops[d]; o = o[(o.seed == 0) & (o.budget == 2)]
    for H in (30, 60):
        t = o[o.H == H].set_index("model")
        first = True
        for m in ["rsrp_now", "joint_margin", "logreg", "xgb", "lstm", "transformer"]:
            r = t.loc[m]
            dd = "--" if m == "rsrp_now" else f"{r.DR - t.loc['rsrp_now'].DR:+.2f} [{r.dDR_lo:+.2f}, {r.dDR_hi:+.2f}]"
            head = (r"\multirow{6}{*}{" + f"{lab}, {H:g}\\,s" + "}") if first else ""
            lines.append(f"{head} & {NAME[m]} & {r.FAR:.2f} [{r.FAR_lo:.2f}, {r.FAR_hi:.2f}] & {r.DR:.2f} [{r.DR_lo:.2f}, {r.DR_hi:.2f}] & {dd} & {r.median_lead:g} \\\\")
            first = False
        lines.append(r"\midrule")
open("tables/t4_paired.tex", "w").write("\n".join(lines[:-1]))

# ---- Table 5: sensitivity (H = 30, seed 0, nominal 2/h)
cfgs = [("r100_c7_m2_w20", "Main"), ("r105_c7_m2_w20", "RSRP $<-105$"), ("r100_c5_m2_w20", "CQI $<5$"),
        ("r100_c7_m3_w20", "$m=3$"), ("r100_c7_m2_w10", "$W=10$"), ("r100_c7_m2_w40", "$W=40$")]
lines = []
for d, lab in DS:
    first = True
    for m in ["rsrp_now", "joint_margin", "logreg", "xgb"]:
        vals = []
        for c, _ in cfgs:
            o = pd.read_csv(f"data/out2_{d}/eval2_ops_{c}.csv"); r = o[(o.H == 30) & (o.budget == 2) & (o.seed == 0) & (o.model == m)]
            vals.append(f"{r.DR.iloc[0]:.2f} ({r.FAR.iloc[0]:.1f})")
        head = (r"\multirow{4}{*}{" + lab + "}") if first else ""; first = False
        lines.append(f"{head} & {NAME[m]} & " + " & ".join(vals) + r" \\")
    lines.append(r"\midrule")
open("tables/t5_sens.tex", "w").write("\n".join(lines[:-1]))

# ---- budget transfer numbers
BT = []
for d, _ in DS:
    o = ops[d].copy(); o["r"] = o.FAR / o.budget; o["D"] = d; BT.append(o)
BT = pd.concat(BT)
N["budget"] = BT.groupby(["D", "model"]).r.agg(median="median", p90=lambda v: np.percentile(v, 90), mx="max").round(2).reset_index().to_dict("records")
N["budget_pooled"] = BT.groupby("D").r.agg(median="median", p90=lambda v: np.percentile(v, 90), mx="max").round(2).to_dict("index")

# ---- Figure 2: DR vs realised FAR
f, ax = plt.subplots(3, 3, figsize=(7.2, 7.4), constrained_layout=True, sharey="row")
for r, (d, lab) in enumerate(DS):
    for c, H in enumerate((10, 30, 60)):
        a = ax[r, c]; a.grid(True)
        for m in SHOW:
            g = ops[d][(ops[d].H == H) & (ops[d].model == m)]
            a.scatter(g.FAR.clip(0.05), g.DR, s=8, color=COL[m], alpha=0.3, marker=MK[m], linewidths=0.8 if m == "random" else 0)
            mu = g.groupby("budget")[["FAR", "DR"]].mean().sort_values("FAR")
            a.plot(mu.FAR.clip(0.05), mu.DR, ls=LS.get(m, "-"), color=COL[m], lw=1.7, marker=MK[m], ms=4, label=NAME[m],
                   markeredgecolor="white" if m != "random" else COL[m], markeredgewidth=0.5)
        a.set_xscale("log"); a.set_xlim(0.08, 20); a.set_xticks([0.1, 0.3, 1, 3, 10]); a.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
        a.set_title(f"{lab}, H = {H} s", loc="left", fontsize=8.5)
    ax[r, 0].set_ylabel("Share of warnable onsets warned")
f.supxlabel("Realised false actions per exposure hour on unseen links (log scale)", fontsize=9)
h, l = ax[0, 0].get_legend_handles_labels(); f.legend(h, l, loc="outside upper center", ncol=4, frameon=False, fontsize=8)
f.savefig("figures/fig2_dr_far.pdf", bbox_inches="tight"); plt.close(f)

# ---- Figure 3: matched-rate paired differences vs RSRP level (link bootstrap, matched_boot.py)
MB = {d: pd.read_csv(f"data/out2_{d}/eval2_matched_boot_{MAIN}.csv") for d, _ in DS}
pd.concat([MB[d].assign(D=d) for d, _ in DS]).to_csv("tables/matched_boot_all.csv", index=False)
models = ["rsrp_trend", "joint_margin", "logreg", "xgb", "lstm", "transformer"]
COL3 = dict(COL, rsrp_trend="#5a5a5a")
f, ax = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True, sharey=True)
for c, (d, lab) in enumerate(DS):
    for rr, H in enumerate((30, 60)):
        a = ax[rr, c]; a.axvline(0, color="0.4", lw=0.8); a.grid(True, axis="x")
        t = MB[d][MB[d].H == H]
        for j, m in enumerate(models):
            for k, rate in enumerate((1, 2, 4)):
                q = t[(t.model == m) & (t.rate == rate)]
                if not len(q) or pd.isna(q.dDR_lo.iloc[0]): continue
                y = j + (k - 1) * 0.26; q = q.iloc[0]
                a.plot([q.dDR_lo, q.dDR_hi], [y, y], color=COL3[m], lw=1.4)
                a.plot(q.dDR, y, marker=["o", "s", "^"][k], color=COL3[m], ms=4, markerfacecolor="white" if k == 0 else COL3[m])
        a.set_yticks(range(len(models))); a.set_yticklabels([NAME[m] for m in models], fontsize=7.5); a.invert_yaxis()
        a.set_title(f"{lab}, H = {H} s", loc="left", fontsize=8.5)
from matplotlib.lines import Line2D
ax[0, -1].legend([Line2D([], [], marker=mk, ls="", color="0.3", markerfacecolor=fc) for mk, fc in (("o", "white"), ("s", "0.3"), ("^", "0.3"))],
                 ["1 false action/h", "2 false actions/h", "4 false actions/h"], frameon=False, fontsize=7, loc="lower right")
f.supxlabel("Difference in share of warnable onsets warned at a matched realised rate, predictor minus RSRP level (95% paired link-bootstrap CI)", fontsize=8)
f.savefig("figures/fig3_matched_diff.pdf", bbox_inches="tight"); plt.close(f)

# ---- Figure 4: AUPRC vs matched detection at 2/h
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
S = pd.concat([acc[d].assign(D=d) for d, _ in DS]).merge(M, on=["D", "H", "model", "seed"])
for c, (d, lab) in enumerate(DS):
    a = ax[c]; a.grid(True)
    g = S[(S.D == d) & (S.H == 30)]
    for m in SHOW:
        q = g[g.model == m]
        a.scatter(q.auprc, q.d2, color=COL[m], marker=MK[m], s=30, edgecolors="white" if m != "random" else None, linewidths=0.6, label=NAME[m], zorder=3)
    a.set_title(f"{lab}, H = 30 s", loc="left", fontsize=8.5); a.set_xlabel("AUPRC (accuracy view)")
ax[0].set_ylabel("Warned at 2 false actions/h")
h, l = ax[0].get_legend_handles_labels(); f.legend(h, l, loc="outside upper center", ncol=4, frameon=False, fontsize=7.5)
f.savefig("figures/fig4_auprc_vs_dr.pdf", bbox_inches="tight"); plt.close(f)

# ---- Figure 5: budget transfer
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True, sharey=True)
for c, (d, lab) in enumerate(DS):
    a = ax[c]; a.axhline(1, color="0.4", lw=0.8); o = BT[BT.D == d]
    for j, m in enumerate(ORDER[1:]):
        v = o[o.model == m].r.values
        a.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), np.clip(v, 0.05, 20), s=5, color="0.35", alpha=0.45, linewidths=0)
        a.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color="#0072B2", lw=1.8)
    a.set_yscale("log"); a.set_ylim(0.05, 20); a.set_yticks([0.1, 0.25, 0.5, 1, 2, 4, 10, 16]); a.set_yticklabels(["0.1", "0.25", "0.5", "1", "2", "4", "10", "16"]); a.minorticks_off()
    a.set_xticks(range(len(ORDER) - 1)); a.set_xticklabels([NAME[m] for m in ORDER[1:]], rotation=60, ha="right", fontsize=7)
    a.set_title(lab, loc="left", fontsize=8.5)
ax[0].set_ylabel("Realised / target false-action rate")
f.savefig("figures/fig5_budget.pdf", bbox_inches="tight"); plt.close(f)

# proxy
N["proxy"] = {d: json.load(open(f"data/out2_{d}/proxy_check.json")) for d in ("cork5g", "cork4g")}
json.dump(N, open("tables/numbers.json", "w"), indent=1, default=float)
print("built")

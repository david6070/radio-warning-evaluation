"""
prep_cork2.py -- UCC (Cork) G-NetTrack traces -> v2 input (one CSV file = one session = one link).
Keeps EVERY row (2G/3G rows have no CQI; net2_data ends the segment there instead of
joining across the gap) and keeps throughput and context for proxy validation.
  python3 prep_cork2.py --root cork/5g --tag 5g --out cork/cork5g_v2.csv
"""
import argparse, glob, os, hashlib
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
files = sorted(f for f in glob.glob(os.path.join(a.root, "**", "*.csv"), recursive=True)
               if "__MACOSX" not in f and not os.path.basename(f).startswith("._"))
parts, man = [], []
for f in files:
    d = pd.read_csv(f, low_memory=False, na_values=["-", ""])
    if not {"Timestamp", "RSRP", "CQI"} <= set(d.columns):
        print("skip (columns):", f); continue
    rel = os.path.relpath(f, a.root).replace(os.sep, "/"); low = rel.lower()
    mob = next((m for m in ("static", "pedestrian", "car", "driving", "bus", "tram", "train") if m in low), "other")
    app = next((m for m in ("download", "netflix", "amazon") if m in low), "other")
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(float("nan"), index=d.index)
    o = pd.DataFrame({"link_id": f"{a.tag}:{rel}",
                      "timestamp": pd.to_datetime(d["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce"),
                      "RSRP": num("RSRP"), "RSRQ": num("RSRQ"), "SINR": num("SNR"), "CQI": num("CQI"),
                      "DL_bitrate": num("DL_bitrate"), "UL_bitrate": num("UL_bitrate"),
                      "network_mode": d.get("NetworkMode"), "operator": d.get("Operatorname"), "cell": d.get("CellID"),
                      "mobility": mob, "app": app})
    o.loc[o.RSRP < -140, "RSRP"] = float("nan")          # -200 is a missing-value code
    parts.append(o.dropna(subset=["timestamp"]))
    man.append(dict(file=rel, rows=len(d), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest()))
df = pd.concat(parts, ignore_index=True); df.to_csv(a.out, index=False)
pd.DataFrame(man).to_csv(a.out.replace(".csv", "_manifest.csv"), index=False)
print(f"{len(files)} files -> {len(df):,} rows, {df.link_id.nunique()} sessions -> {a.out}")
print("mobility:", df.groupby("mobility").link_id.nunique().to_dict(), "| app:", df.groupby("app").link_id.nunique().to_dict())
print("operators:", df.groupby("operator").link_id.nunique().to_dict())
print("years:", sorted(df.timestamp.dt.year.dropna().unique().tolist()), "| modes:", df.network_mode.value_counts().head(6).to_dict())

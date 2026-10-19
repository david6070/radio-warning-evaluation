"""
prep_manaus2.py -- raw Mendeley/Amazonas Manaus file -> v2 input.
Link = device Model x serving LTE cell (CI LTE): the anonymised file has no
device identifier, so measurements of several phones of one model on one cell
can be interleaved (stated as a limitation; Manaus is exploratory in v2).
Rows with missing RSRP/CQI are KEPT (net2_data ends a segment at them). No
minimum-session filter (short links simply yield no eligible samples).
  python3 prep_manaus2.py --input df_anonymizedManaus.csv --out manaus_v2.csv
"""
import argparse, hashlib
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
raw = pd.read_csv(a.input, low_memory=False)
model = raw["Model"].astype(str).str.strip().str.replace(" ", "_")
ci = pd.to_numeric(raw["CI LTE"], errors="coerce")
df = pd.DataFrame({"link_id": model + "_" + ci.astype("Int64").astype(str),
                   "timestamp": pd.to_datetime(raw["Current Date Time"], errors="coerce"),
                   "RSRP": pd.to_numeric(raw["RSRP LTE"], errors="coerce"), "RSRQ": pd.to_numeric(raw["RSRQ LTE"], errors="coerce"),
                   "SINR": pd.to_numeric(raw["SNR LTE"], errors="coerce"), "CQI": pd.to_numeric(raw["CQI LTE"], errors="coerce"),
                   "network_type": raw.get("Network Type"), "operator": raw.get("Network Operator")})
df = df[ci.notna() & df.timestamp.notna()]
df.to_csv(a.out, index=False)
print(f"raw rows {len(raw):,} -> {len(df):,} rows with a cell id and timestamp, {df.link_id.nunique()} links -> {a.out}")
print("input sha256:", hashlib.sha256(open(a.input, "rb").read()).hexdigest())

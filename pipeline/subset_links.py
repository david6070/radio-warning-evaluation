"""
subset_links.py -- keep the sessions of one application (Cork prepared files).
  python3 subset_links.py --input cork5g_v2.csv --app download --out cork5g_dl.csv
Prints the number of links and rows kept, and how often DL_bitrate is missing.
"""
import argparse, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--app", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
df = pd.read_csv(a.input, low_memory=False)
keep = df[df["app"] == a.app]
keep.to_csv(a.out, index=False)
dl = pd.to_numeric(keep["DL_bitrate"], errors="coerce")
print(f"{a.app}: {keep.link_id.nunique()} links, {len(keep):,} rows; DL_bitrate missing {dl.isna().mean():.1%}, "
      f"median {dl.median():.0f} kbit/s, below 1 Mbit/s {(dl < 1000).mean():.1%}")

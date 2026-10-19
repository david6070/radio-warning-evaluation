"""dump_manifests.py -- write the split manifest and fold info stored in every score file to JSON,
plus SHA-256 hashes of score files and input CSVs (provenance for the release).
  python3 dump_manifests.py --scores out2_cork5g --input cork/cork5g_v2.csv --out release/manifests_cork5g"""
import argparse, glob, hashlib, json, os, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True); ap.add_argument("--input", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
index = {"input": a.input, "input_sha256": sha(a.input), "score_files": {}}
for f in sorted(glob.glob(f"{a.scores}/s2_*.npz")):
    z = np.load(f, allow_pickle=True); info = json.loads(str(z["info"]))
    json.dump(info, open(os.path.join(a.out, os.path.basename(f).replace(".npz", ".json")), "w"), indent=1)
    index["score_files"][os.path.basename(f)] = dict(sha256=sha(f), n_rows=int(len(z["rows_all"])))
json.dump(index, open(os.path.join(a.out, "INDEX.json"), "w"), indent=1)
print(f"{len(index['score_files'])} manifests -> {a.out}")

#!/bin/bash
# Builds release/ with code, environment, pre-training check logs, split manifests and all evaluation outputs.
cd /work/dod220001/netrev && source /work/dod220001/tmlcn/venv/bin/activate
mkdir -p release/code release/checks
cp juno_net2/*.py juno_net2/*.sbatch juno_net2/tasks2_*.txt juno_net2/README_V2.md release/code/
python3 --version > release/environment.txt 2>&1; pip freeze >> release/environment.txt 2>/dev/null
cp cork/*_manifest.csv release/ 2>/dev/null
for D in manaus:manaus/manaus_v2.csv cork5g:cork/cork5g_v2.csv cork4g:cork/cork4g_v2.csv; do
  N=${D%%:*}; F=${D##*:}
  (cd juno_net2 && python3 test_v2.py --input ../$F) > release/checks/checks_$N.txt 2>&1
  python3 juno_net2/dump_manifests.py --scores out2_$N --input $F --out release/manifests_$N
  mkdir -p release/results_$N && cp out2_$N/eval2_* out2_$N/proxy_check.json release/results_$N/ 2>/dev/null
done
sha256sum /work/dod220001/tmlcn/df_anonymizedManaus.csv cork/LTE_Dataset.zip cork/5g_repo/5G-production-dataset.zip > release/raw_data_sha256.txt 2>/dev/null
grep -h "CHECKS" release/checks/*.txt
tar czf release_v2.tgz release && ls -la release_v2.tgz

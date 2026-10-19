#!/bin/bash
# v2 evaluation for one dataset (main configuration at H=10,30,60 + sensitivity configurations at H=30).
#   cd /work/dod220001/netrev && sbatch --chdir=/work/dod220001/netrev --export=ALL,INPUT=<csv>,OUT=<dir> juno_net2/net2_eval.sbatch
#SBATCH --job-name=net2_eval
#SBATCH --output=logs/net2_eval_%j.out
#SBATCH --error=logs/net2_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=normal
module load gnu14 2>/dev/null || true
source /work/dod220001/tmlcn/venv/bin/activate
cd juno_net2
python3 -u net2_eval.py --input "$INPUT" --scores "../$OUT" --H 10 30 60 --B 1000
for c in "-105 7 2 20" "-100 5 2 20" "-100 7 3 20" "-100 7 2 10" "-100 7 2 40"; do
  read RT CT MR W <<< "$c"
  python3 -u net2_eval.py --input "$INPUT" --scores "../$OUT" --H 30 --B 0 --rsrp_thr $RT --cqi_thr $CT --min_run $MR --W $W
done
python3 -u matched_boot.py --perlink "../$OUT/eval2_perlink_r100_c7_m2_w20.csv.gz" --out "../$OUT" --rates 1 2 3 4 --B 2000
case "$INPUT" in *cork*) python3 -u proxy_check.py --input "$INPUT" --out "../$OUT" ;; esac
echo "done $(date)"

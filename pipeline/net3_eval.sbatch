#!/bin/bash
# Extension evaluation for one output directory.
#   EVENT != rc : full evaluation of the new event (net2_eval.py), then the A2 rule, then matched-rate bootstrap
#   EVENT == rc : existing main results; adds only the A2 rule and the comparisons against RSRP level and RSRP trend
# --export=ALL,INPUT=<csv>,OUT=<dir>,EVENT=..,MR=..
#SBATCH --job-name=net3_eval
#SBATCH --output=logs/net3_eval_%j.out
#SBATCH --error=logs/net3_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=normal
module load gnu14 2>/dev/null || true
source /work/dod220001/tmlcn/venv/bin/activate
set -e
cd juno_net2
E="--event $EVENT --min_run $MR"
TAG=$(python3 -c "import net2_data as N; print(N.Cfg(event='$EVENT', min_run=$MR).tag())")
echo "$INPUT $OUT tag=$TAG $(date)"
if [ "$EVENT" != "rc" ]; then
  python3 -u net2_eval.py --input "$INPUT" --scores "../$OUT" --H 10 30 60 --B 1000 $E
fi
python3 -u net3_a2.py --input "$INPUT" --out "../$OUT" --H 10 30 60 $E
PL="../$OUT/eval2_perlink_${TAG}.csv.gz ../$OUT/eval3_a2_perlink_${TAG}.csv.gz"
python3 -u matched_boot.py --perlink $PL --out "../$OUT" --rates 1 2 3 4 --B 2000 --ref rsrp_now   --name _ext_ref_level
python3 -u matched_boot.py --perlink $PL --out "../$OUT" --rates 1 2 3 4 --B 2000 --ref rsrp_trend --name _ext_ref_trend
echo "done $(date)"

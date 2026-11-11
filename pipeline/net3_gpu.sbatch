#!/bin/bash
# Extension (review of 11 Nov 2026): train the predictors for one dataset and one event definition.
# Submitted by run_ext.sh with --export=ALL,INPUT=<csv>,OUT=<dir>,EVENT=<rc|tput_floor|tput_drop>,MR=<min_run>,MODELS="<list>"
#SBATCH --job-name=net3_gpu
#SBATCH --output=logs/net3_gpu_%j.out
#SBATCH --error=logs/net3_gpu_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=h100,a30,h200
#SBATCH --gres=gpu:1
module load gnu14 2>/dev/null || true
source /work/dod220001/tmlcn/venv/bin/activate
set -e
echo "$INPUT -> $OUT event=$EVENT min_run=$MR models=[$MODELS] on $(hostname) $(date)"
cd juno_net2
for H in 10 30 60; do
  for M in $MODELS; do
    case $M in xgb|lstm|transformer) SEEDS="0 1 2";; *) SEEDS="0";; esac
    for S in $SEEDS; do
      python3 -u net2_train.py --input "$INPUT" --model "$M" --H "$H" --seed "$S" --out "../$OUT" --event "$EVENT" --min_run "$MR"
    done
  done
done
echo "done $(date)"

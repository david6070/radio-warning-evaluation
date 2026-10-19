#!/bin/bash
# Re-evaluation of the main configuration with per-link output + matched-rate bootstrap (no retraining).
#   cd /work/dod220001/netrev && sbatch --chdir=/work/dod220001/netrev --export=ALL,INPUT=<csv>,OUT=<dir> juno_net2/net2_eval_main.sbatch
#SBATCH --job-name=net2_evm
#SBATCH --output=logs/net2_evm_%j.out
#SBATCH --error=logs/net2_evm_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=normal
module load gnu14 2>/dev/null || true
source /work/dod220001/tmlcn/venv/bin/activate
cd juno_net2
python3 -u net2_eval.py --input "$INPUT" --scores "../$OUT" --H 10 30 60 --B 1000
python3 -u matched_boot.py --perlink "../$OUT/eval2_perlink_r100_c7_m2_w20.csv.gz" --out "../$OUT" --rates 1 2 4 --B 2000
echo "done $(date)"

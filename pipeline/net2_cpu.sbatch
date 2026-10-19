#!/bin/bash
# v2 training: one array task = one line of juno_net2/tasks2_cpu.txt (model H seed rsrp_thr cqi_thr min_run W).
#   cd /work/dod220001/netrev && sbatch --chdir=/work/dod220001/netrev --export=ALL,INPUT=<csv>,OUT=<dir> --array=1-$(wc -l < juno_net2/tasks2_cpu.txt) juno_net2/net2_cpu.sbatch
#SBATCH --job-name=net2_cpu
#SBATCH --output=logs/net2_cpu_%A_%a.out
#SBATCH --error=logs/net2_cpu_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=normal

module load gnu14 2>/dev/null || true
source /work/dod220001/tmlcn/venv/bin/activate
read MODEL H SEED RT CT MR W <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" juno_net2/tasks2_cpu.txt)"
echo "task ${SLURM_ARRAY_TASK_ID}: $MODEL H=$H seed=$SEED cfg=$RT/$CT/$MR/$W on $(hostname) $(date)"
cd juno_net2 && python3 -u net2_train.py --input "$INPUT" --model "$MODEL" --H "$H" --seed "$SEED" --out "../$OUT" --rsrp_thr "$RT" --cqi_thr "$CT" --min_run "$MR" --W "$W"
echo "done $(date)"

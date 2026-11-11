#!/bin/bash
# Submits the extension runs requested in the review of 11 Nov 2026.
#   (a) throughput-defined events (independent of RSRP/CQI) on Cork 5G NSA download sessions and Cork 4G
#   (b) matched-rate comparisons against the RSRP trend rule (in addition to RSRP level)
#   (c) an A2-like serving-cell rule with time-to-trigger and hysteresis
# Usage (on Juno, after copying the new pipeline files into juno_net2/):
#   cd /work/dod220001/netrev && bash juno_net2/run_ext.sh          # CPU predictors only
#   GPU=1 bash juno_net2/run_ext.sh                                  # also LSTM and Transformer for (a)
set -e
cd /work/dod220001/netrev
mkdir -p logs
C5=cork/cork5g_v2.csv; C4=cork/cork4g_v2.csv; MA=manaus/manaus_v2.csv
for f in $C5 $C4 $MA out2_cork5g/eval2_perlink_r100_c7_m2_w20.csv.gz out2_cork4g/eval2_perlink_r100_c7_m2_w20.csv.gz out2_manaus/eval2_perlink_r100_c7_m2_w20.csv.gz; do
  [ -f "$f" ] || { echo "missing $f -- fix the path at the top of run_ext.sh"; exit 1; }
done
source /work/dod220001/tmlcn/venv/bin/activate
(cd juno_net2 && python3 net3_a2.py --input x --out x --selftest && python3 test_full_replay.py | tail -1)
python3 juno_net2/subset_links.py --input $C5 --app download --out cork/cork5g_dl.csv
X="--chdir=/work/dod220001/netrev"
CPU_MODELS="rsrp_now rsrp_trend rsrp_ewma joint_margin random logreg xgb"
# (a) throughput-defined events
for DS in "cork5g_dl:cork/cork5g_dl.csv" "cork4g:$C4"; do
  N=${DS%%:*}; F=${DS##*:}
  for EV in tput_floor tput_drop; do
    OUT=out3_${N}_${EV}; mkdir -p $OUT
    J=$(sbatch --parsable $X --export=ALL,INPUT=../$F,OUT=$OUT,EVENT=$EV,MR=3,MODELS="$CPU_MODELS" juno_net2/net3_train.sbatch)
    DEP="afterok:$J"
    if [ "${GPU:-0}" = "1" ]; then
      G=$(sbatch --parsable $X --export=ALL,INPUT=../$F,OUT=$OUT,EVENT=$EV,MR=3,MODELS="lstm transformer" juno_net2/net3_gpu.sbatch)
      DEP="$DEP:$G"
    fi
    sbatch $X --dependency=$DEP --export=ALL,INPUT=../$F,OUT=$OUT,EVENT=$EV,MR=3 juno_net2/net3_eval.sbatch
    echo "submitted $N $EV -> $OUT"
  done
done
# (b) + (c) on the main event, reusing the existing scores (no retraining)
for DS in "cork5g:$C5" "cork4g:$C4" "manaus:$MA"; do
  N=${DS%%:*}; F=${DS##*:}
  sbatch $X --export=ALL,INPUT=../$F,OUT=out2_$N,EVENT=rc,MR=2 juno_net2/net3_eval.sbatch
  echo "submitted main-event A2 + trend comparison for $N"
done
squeue -u $USER

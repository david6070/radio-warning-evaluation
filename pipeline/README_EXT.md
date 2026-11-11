# Extension runs (review of 11 November 2026)

| | What | Retraining? | Script |
|---|---|---|---|
| (a) | Throughput-defined events, independent of RSRP and CQI, on Cork 5G NSA **download** sessions and on Cork 4G | yes (CPU; GPU optional) | `net3_train.sbatch`, `net3_gpu.sbatch`, `net3_eval.sbatch` |
| (b) | Matched-rate comparisons against the **RSRP trend** rule as well as RSRP level | no | `net3_eval.sbatch` (EVENT=rc) |
| (c) | **A2-like rule**: serving-cell RSRP with time-to-trigger k samples and hysteresis h dB, threshold calibrated like every other predictor | no | `net3_a2.py` |

## Event definitions for (a) (`net2_data.Cfg.event`)
* `tput_floor`: downlink throughput `DL_bitrate` < 1000 kbit/s (1 Mbit/s).
* `tput_drop`: `DL_bitrate` < 0.5 x median of the previous 10 measurements in the segment.
Both use min_run = 3 (a degradation must last >= 3 measurements), min_clear = 3, as the
radio event otherwise. Inputs to every predictor are unchanged (radio measurements only),
so no rule is defined on the same quantity as the event. The main radio event (`rc`) and
its output tags are unchanged (`r100_c7_m2_w20`).

## A2-like rule variants (`a2_k{k}_h{h}`)
k:h = 3:2 (main), 1:2, 5:2, 3:0. Enter when RSRP + h < theta for the last k measurements,
leave when RSRP - h > theta; theta is calibrated per outer fold on the calibration links
(seed-0 split of net2_train.py) for every budget.

## Steps on Juno
```bash
cd /work/dod220001/netrev
cp -r juno_net2 juno_net2_bak_20261111          # keep the version that produced the paper
cp juno_ext/* juno_net2/                          # new and updated files
bash juno_net2/run_ext.sh                         # CPU only; or: GPU=1 bash juno_net2/run_ext.sh
squeue -u dod220001                               # 4 training + 4 evaluation jobs (a), 3 evaluation jobs (b, c)
# when all jobs have finished:
bash juno_net2/collect_ext.sh                     # -> ext_results.tgz
```
`run_ext.sh` first runs the two self-tests (A2 rule, full causal replay) and stops if a
prepared input or a main per-link file is missing.

Outputs: `out3_{cork5g_dl,cork4g}_{tput_floor,tput_drop}/` (same files as the main study)
and, in each `out2_*`, `eval3_a2_*` plus `eval2_matched_boot_*_ext_ref_{level,trend}.csv`.
The main results files are not overwritten.

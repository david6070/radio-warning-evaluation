# Radio degradation warning: evaluation as preventive actions under a false-alarm budget

Code and results for the paper

> D. O. Daniel, *Warning before the link degrades: evaluating radio-quality forecasts as
> preventive actions under a false-alarm budget*, submitted to *Computer Networks* (2026).

The study forecasts radio-link degradation onsets (RSRP < -100 dBm and CQI < 7 for >= 2
samples) from the past 20 measurements and evaluates warnings **offline** as preventive
actions: an action is valid for a horizon H, is charged when issued or renewed, and is
judged by the share of warnable onsets it covers and by false actions per exposure hour,
on links used neither for fitting nor for choosing the alarm threshold.

## Contents
| Folder | What it is |
|---|---|
| `pipeline/` | Data preparation, training (nested fit / early-stop / calibration / test splits), action-based evaluator, pre-training checks, matched-rate paired bootstrap, Slurm job scripts and task lists |
| `paper_build/` | `build_v2.py` regenerates every table (LaTeX fragments in `tables/`) and Figures 2-5 from the evaluation outputs in `data/out2_*` |
| `evidence/` | Pre-training check logs for each dataset, SHA-256 of the raw data used, and the audit showing that the full-grid threshold search changed none of the 1,071 main operating points |
| Release asset `release_v2.tgz` (v1.0) | Complete record: final code, environment (`pip freeze`), split manifests of all 213 training runs, per-link evaluation counts (from which every operating point and bootstrap interval can be recomputed), check logs |

## Data (not redistributed)
- Cork 5G: Raca et al., MMSys 2020 -- https://github.com/uccmisl/5Gdataset (`5G-production-dataset.zip`)
- Cork 4G: Raca et al., MMSys 2018 -- https://zenodo.org/records/1219679 (`LTE_Dataset.zip`)
- Manaus: Silva et al., Data in Brief 57 (2024) 111146 -- Mendeley Data, file `df_anonymizedManaus.csv`

SHA-256 of the files used are in `evidence/checks/raw_data_sha256.txt`.

## Reproducing
```bash
# 1. prepare inputs
python3 pipeline/prep_cork2.py   --root <unzipped 5G dataset> --tag 5g --out cork5g_v2.csv
python3 pipeline/prep_cork2.py   --root <unzipped 4G dataset> --tag 4g --out cork4g_v2.csv
python3 pipeline/prep_manaus2.py --input df_anonymizedManaus.csv --out manaus_v2.csv
# 2. checks that must pass before training (always-on/off, oracle, causality, segment boundary, full threshold grid)
cd pipeline && python3 test_v2.py --input ../cork5g_v2.csv
# 3. train every line of tasks2_cpu.txt / tasks2_gpu.txt (model H seed rsrp_thr cqi_thr min_run W), e.g.
python3 net2_train.py --input ../cork5g_v2.csv --model xgb --H 30 --seed 0 --out ../out2_cork5g
# 4. evaluate, then matched-rate bootstrap
python3 net2_eval.py --input ../cork5g_v2.csv --scores ../out2_cork5g --H 10 30 60 --B 1000
python3 matched_boot.py --perlink ../out2_cork5g/eval2_perlink_r100_c7_m2_w20.csv.gz --out ../out2_cork5g --rates 1 2 3 4 --B 2000
# 5. tables and figures
cd ../paper_build && python3 build_v2.py
```
The Slurm scripts (`pipeline/net2_*.sbatch`) run steps 3-4 as array jobs; paths in them refer to the
UT Dallas Juno cluster and need adapting. Python 3.12 with pandas, numpy, scikit-learn, xgboost,
PyTorch, scipy and matplotlib; exact versions are in the release asset (`environment.txt`).

## License
Code: MIT (see `LICENSE`). The datasets belong to their authors and are used under their own licences.

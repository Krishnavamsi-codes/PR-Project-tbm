# Earthquake Building Damage Classification (Pattern Recognition project)

Dataset: Richter's Predictor, Nepal Gorkha 2015 (260,601 buildings, 38 features, 3 damage grades).
Code repository: https://github.com/Krishnavamsi-codes/PR-Project-tbm (keep public until grades are awarded)

## Run everything (no intervention)
```bash
pip install -r requirements.txt
python code/main.py            # full run: 20k stratified subsample, 5-fold x 3 repeats
python code/main.py --quick    # small smoke test
```
`code/main.py` is the equivalent of `Main.m`: it runs EDA, all three tracks, and writes every table and
figure to `results/` and `figures/`. Finished experiments are cached in `results/*.json`
(delete them or pass `--force` to recompute). Needs `data/train_values.csv` and `data/train_labels.csv`.

## Protocol
- Fixed-seed (42) class-proportional subsample of 20,000 rows; stratified 5-fold CV x 3 repeats = 15 runs.
- Every learned step (target encoding, scaling, one-hot, PCA, LDA, feature selection, class-weight tuning)
  lives in an sklearn Pipeline that is fitted on the training fold only. Target encoding is out-of-fold inside
  the training fold. No test row is ever used for fitting.
- Metrics per run: accuracy, macro-F1, per-class precision/recall, confusion matrix; reported as mean ± std.

## Layout
| File | Owner | Content |
|---|---|---|
| `code/common.py`, `eda.py`, `track1_baselines.py`, `main.py`, `report.py` | Person 1 | data, CV harness, metrics, EDA, raw baselines, integration |
| `code/track2_features.py` | Person 2 | target encoding, engineered features, PCA, LDA, MI, Fisher, ablation |
| `code/track3_methods.py` | Person 3 | cost-sensitive Bayes, Mahalanobis kNN, balanced kNN, ensemble, MLP (reference), significance |

## Scale study (20k vs 260k)
```bash
python code/scale_study.py       # subsample-vs-full distributions + learning curves -> results/scale_study.json
python code/importance_study.py  # random-forest importances, 20k vs all rows
python code/error_study.py       # confusion matrices and per-region gains, 20k vs all rows
```

# Forecasting One-Season Football Player Development

COMP1818 (Artificial Intelligence Applications) coursework. Predicts attackers'
next-season non-penalty xG + xA per 90 minutes from the previous season's
statistics. Compares a no-change baseline, Linear Regression, Random Forest and
XGBoost, and tests global vs position-specific models.

## Data
Understat player statistics (2014/15 to 2023/24) via the Kaggle dataset
"Understat-data" by Cody Tipton:
https://www.kaggle.com/datasets/codytipton/understat-data

The raw data is **not** redistributed here. Download it, extract
`player_stats.csv` and place it in `dataraw/`. See the Kaggle page for the
licence. The partial 2024/25 season in the dataset is excluded.

## Setup
    pip install -r requirements.txt

## Reproduce
Run from the repository root:

    python dataraw/build_dataset.py
    python dataraw/train_eval.py val
    python dataraw/train_eval.py analysis
    python dataraw/train_eval.py trend val
    python dataraw/train_eval.py test
    python dataraw/train_eval.py trend test
    python dataraw/train_eval.py testpaired
    python dataraw/train_eval.py save

Run the test commands only once, after tuning on validation.
Seeds are fixed, so results should match the saved outputs in `results/`
(small differences are possible with different library versions).
Trained models are in `processed/models/`; tuned hyperparameters are in
`processed/best_params.json`.

## Split (by input season)
- Train: 2014 to 2020
- Validation: 2021 (target 2022/23)
- Test: 2022 (target 2023/24)

## Files
- `dataraw/build_dataset.py`: cleaning, position groups, per-90 features, season pairing
- `dataraw/train_eval.py`: tuning, test evaluation, paired CIs, trend experiment, model saving
- `dataraw/check_seasons.py`, `dataraw/build_pairs_check.py`: data checks
- `dataraw/make_figure.py`: draws Figure 1 (paired MAE change vs baseline) from `results/test_paired_output.txt`; output in `figures/`
- `results/`: saved outputs of each run

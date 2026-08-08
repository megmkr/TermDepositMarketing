# Term Deposit Marketing

Predicting whether a customer will subscribe to a term deposit from call-center marketing data, so outreach can be targeted at the customers most likely to convert instead of the full call list.

## Problem

The dataset (`data/raw/term-deposit-marketing-2020.csv`) contains 40,000 calls made during a term deposit marketing campaign, with 13 features about the customer (age, job, marital status, education, existing loans, ...) and the call itself (contact method, day/month, duration, number of contacts), plus the outcome `y`: whether the customer subscribed.

Only **7.2% of calls result in a subscription**, so this is a heavily imbalanced binary classification problem. The interesting part is separating the minority of likely subscribers from the majority who aren't.

## Approach

1. **EDA & feature engineering** (`notebooks/feature_engineering.ipynb`) — encode categorical/binary fields, engineer features from call timing and contact frequency (`quarter`, `season`, `campaign_squared`, `many_contacts`, `long_call`, `duration_log`), and compare encoding strategies (ordinal vs. dummy).
2. **Model selection & tuning** (`notebooks/modeling.ipynb`) — compare Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and XGBoost across encodings and resampling strategies (SMOTE vs. random undersampling), then tune the strongest candidates with `GridSearchCV` (scoring on F1, since accuracy is a poor metric at 7% positive class).
3. **Reusable pipeline** (`src/`) — the same feature engineering and model configs, packaged so training/evaluation/plotting can be re-run from the command line instead of re-executing notebook cells.

## Project layout

```
├── data/
│   ├── raw/            term-deposit-marketing-2020.csv (source data)
│   ├── interim/         intermediate encoding experiments
│   ├── processed/        train/test splits ready for modeling
│   └── external/
├── notebooks/
│   ├── feature_engineering.ipynb   EDA, encoding, feature engineering
│   └── modeling.ipynb              model comparison, tuning, threshold selection
├── src/
│   ├── config.py        paths, classifiers, and param grids for all 6 models
│   ├── dataset.py        load / split / scale / resample
│   ├── features.py       feature engineering + encoding (mirrors the notebook)
│   ├── train.py          CLI entry point: train, evaluate, or plot
│   ├── predict.py        load a saved model and score it
│   └── plots.py          correlation matrix, feature importance, confusion matrix,
│                          ROC/PR curves, F1 comparison across models
├── models/               trained model pickles (one per algorithm)
├── reports/figures/      generated plots
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run from the repository root so the relative data/model/figure paths resolve correctly:

```bash
# train all 6 models (rf, lr, knn, svm, xgb, dt) with grid search, saving to models/
python src/train.py --mode train

# load the saved models and print F1 score for each
python src/train.py --mode evaluate

# generate correlation matrix, per-model feature importance / confusion matrices,
# an ROC + precision-recall comparison, and an F1 comparison chart into reports/figures/
python src/train.py --mode plot
```

Note: the `svm` and `xgb` grid searches are the slowest of the six.

## Results

Random Forest and XGBoost were the strongest performers during model comparison; `--mode plot` generates dedicated feature importance, confusion matrix, and ROC/precision-recall plots for both (`reports/figures/rf_*.png`, `reports/figures/xgb_*.png`, `reports/figures/model_comparison_curves.png`), plus an F1 comparison across all six models (`reports/figures/f1_comparison.png`).

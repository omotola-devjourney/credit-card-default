# Credit Card Default Prediction

Predicts whether a customer will default next month using three classic ML models: Logistic Regression (scaled), Random Forest, and XGBoost.

## Dataset
- UCI “Default of Credit Card Clients”
- Target: `default payment next month` (0/1)

## Method (short)
- Train/test split with `stratify` and fixed seed.
- **Logistic Regression** on **normalized** numeric features + one-hot categorical; `class_weight="balanced"`.
- **Random Forest** and **XGBoost** on **raw** (unnormalized) features; RF uses `class_weight="balanced_subsample"`, XGB uses `scale_pos_weight` and `random_state=42`.

## Results (test set)
| Model                | Accuracy | Recall (class 1) | ROC-AUC | Predicted positive rate |
|---------------------|---------:|-----------------:|--------:|------------------------:|
| Logistic Regression | **0.682** | **0.637**        | **0.717** | **0.379** |
| Random Forest       | **0.813** | **0.422**        | **0.771** | **0.153** |
| XGBoost             | **0.763** | **0.601**        | **0.773** | **0.282** |

**Why the predicted-positive rates differ:** all three models use the default threshold of 0.5, but their score distributions are calibrated differently and we also applied imbalance handling (e.g., `class_weight` and `scale_pos_weight`). As a result, Logistic Regression flags ~38% as positive, XGBoost ~28%, and Random Forest ~15%. This is expected; if your cost of missing a defaulter is high, you’d lower the threshold to improve recall at the expense of more false positives.

## How to run (locally)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# open and run the notebook (or your scripts)
```

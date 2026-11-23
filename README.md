# Lab 3: Ticket Escalation Classification

Binary classification on a synthetic service-ticket dataset (`escalated` = 1 means the ticket was escalated). The focus is on choosing metrics and decision thresholds based on the cost of errors, not only on getting a high score.

## Contents

| File | What it is |
|---|---|
| `escalation_classification.ipynb` | Completed notebook with executed outputs |
| `classification_escalation.csv` | Dataset (900 rows) |
| `RECOMMENDATION.md` | 300-500 word recommendation (also at the end of the notebook) |
| `figures/` | Confusion matrix, ROC curve, threshold trade-off, k-NN CV plot |
| `requirements.txt` | Python packages used |

## What the notebook does

1. Loads the data and checks shape, types, missing values and class balance (about 20.7% escalated)
2. Splits into train/test (`test_size=0.25`, `random_state=42`, stratified)
3. Builds preprocessing pipelines (median impute + scale for numeric, most-frequent impute + one-hot for categorical)
4. Compares a majority baseline, Logistic Regression, LDA and k-NN (k=15)
5. Plots the confusion matrix and ROC curve for Logistic Regression
6. Evaluates thresholds 0.20 to 0.70 using precision, recall, F1, FP, FN and the cost `5*FN + FP`
7. Tunes k for k-NN with stratified 5-fold CV (F1)
8. Explains prior / likelihood / posterior / evidence and derives the cost-sensitive threshold `1/(1+5) = 0.167`
9. Gives a final recommendation

## Main results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority baseline | 0.796 | 0.00 | 0.000 | 0.000 | n/a |
| Logistic Regression | 0.822 | 0.75 | 0.196 | 0.310 | 0.811 |
| LDA | 0.822 | 0.75 | 0.196 | 0.310 | 0.809 |
| k-NN (k=15) | 0.796 | 0.50 | 0.087 | 0.148 | 0.705 |

At the default 0.50 threshold the logistic regression cost is 188; at 0.20 it falls to 98.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook escalation_classification.ipynb
```

Keep the notebook in the same folder as the CSV.

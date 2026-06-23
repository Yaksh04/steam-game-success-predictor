# Classifier Experiments

## Logistic Regression Baseline

Features:
22 launch-time features

Accuracy:
0.511

Macro F1:
0.428

Observations:

- Strong performance identifying limited reach games
- High recall for breakout titles
- Middle market tiers are hardest due to natural overlap

## Initial Model Comparison

| Model               | Accuracy | Macro F1 |
| ------------------- | -------- | -------- |
| Logistic Regression | 0.511    | 0.428    |
| Random Forest       | 0.544    | 0.470    |
| XGBoost             | 0.611    | 0.442    |

Observation:

XGBoost achieved highest accuracy but favored the majority class.
Macro F1 selected Random Forest because balanced tier prediction is more important.

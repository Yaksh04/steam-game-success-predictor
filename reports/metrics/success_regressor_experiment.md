# GameScope — Ordinal Success Regression Experiment Report

## Objective

The goal of this experiment was to evaluate whether Steam game success prediction could be improved by treating commercial success as a continuous ordinal problem instead of a direct multi-class classification problem.

The existing classifier predicts:

| Class | Meaning        |
| ----- | -------------- |
| 0     | Limited Reach  |
| 1     | Niche Success  |
| 2     | Market success |
| 3     | Breakout Hit   |

However, Steam success is naturally continuous:

```
Small indie game
        ↓
Moderate success
        ↓
Popular title
        ↓
Major breakout hit
```

A game barely entering the "Successful" tier is closer to a high-performing "Niche Success" game than a top AAA hit.

Therefore, a regression-based approach was tested.

---

# 1. Motivation

## Problem With Pure Classification

The success classes are ordered:

```
Limited Reach < Niche Success < Successful < Breakout Hit
```

A normal classifier treats all mistakes equally.

Example:

Prediction A:

```
Actual:
Breakout Hit

Predicted:
Successful
```

Prediction B:

```
Actual:
Breakout Hit

Predicted:
Limited Reach
```

Both count as incorrect, even though Prediction A is much closer.

---

## Proposed Solution

Instead of directly predicting success tiers:

```
Features
    |
    ↓
Success Tier
```

predict a continuous market success percentile:

```
Features
    |
    ↓
Owner Percentile (0-1)
    |
    ↓
Convert into Success Tier
```

This allows the model to learn relative success ranking.

---

# 2. Target Engineering

The regression target used:

```
owner_percentile
```

Creation process:

1. Convert SteamSpy owner ranges into midpoint estimates.

Example:

```
200,000 - 500,000 owners

↓

350,000 estimated owners
```

2. Convert owner midpoint values into percentile ranks.

Example:

```
Top performing games → close to 1.0

Low reach games → close to 0.0
```

This avoids issues caused by Steam's extremely skewed ownership distribution.

---

# 3. Leakage Prevention

A strict leakage audit was maintained.

The following post-launch features were excluded:

| Feature          | Reason                        |
| ---------------- | ----------------------------- |
| estimated_owners | Used to create target         |
| owners_midpoint  | Direct popularity information |
| positive reviews | Only available after launch   |
| negative reviews | Only available after launch   |
| review count     | Direct reception signal       |

`owner_percentile` was used only as the regression target.

It was never included as a model input feature.

---

# 4. Model Training

Dataset:

```
Total samples: 44,500
Features: 122
```

Model objective:

Predict:

```
owner_percentile
```

Evaluation metrics:

- Mean Absolute Error (MAE)
- R² Score
- Converted tier Macro F1 Score

---

# 5. Regression Performance

## Continuous Prediction Results

| Metric   | Score |
| -------- | ----- |
| MAE      | 0.162 |
| R² Score | 0.422 |

## Interpretation

The model explains approximately:

```
42% of variance
```

in Steam market success percentile using only launch-time information.

This shows that metadata features contain meaningful predictive signals.

---

# 6. Converted Success Tier Performance

After predicting owner percentile, predictions were converted back into the four success tiers.

Classification results:

| Class         | Precision | Recall | F1   |
| ------------- | --------- | ------ | ---- |
| Limited Reach | 0.82      | 0.58   | 0.68 |
| Niche Success | 0.26      | 0.67   | 0.37 |
| Successful    | 0.39      | 0.25   | 0.31 |
| Breakout Hit  | 0.90      | 0.27   | 0.41 |

Overall:

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.50  |
| Macro F1 | 0.442 |

---

# 7. Comparison With Direct Classifier

## Tuned XGBoost Classifier

| Model                     | Accuracy | Macro F1 |
| ------------------------- | -------- | -------- |
| Tuned XGBoost Classifier  | 0.600    | 0.532    |
| Ordinal Success Regressor | 0.500    | 0.442    |

The direct classifier performs better for predicting discrete success tiers.

Difference:

```
Macro F1 decrease:

0.532 → 0.442
```

---

# 8. Analysis

## Strengths

The ordinal regressor performs well at ranking overall success.

Evidence:

```
R² = 0.422
```

This indicates the model learned meaningful market signals.

Important predictive information includes:

- game category
- Steam tags
- genres
- pricing
- platform availability
- developer reputation
- publisher reputation

---

## Weakness

Regression predictions are conservative.

Example:

Actual:

```
Owner percentile:
0.95
```

Predicted:

```
Owner percentile:
0.78
```

For regression:

This is a small error.

For classification:

```
Breakout Hit

↓

Successful
```

This becomes a full class mistake.

As a result:

- Breakout predictions have high precision
- Many actual breakout games are missed

---

# 9. Final Architecture Decision

The ordinal regressor will not replace the success classifier.

Instead, both models will be used for different purposes.

Final GameScope architecture:

```
                  Game Metadata

                         |

          --------------------------------

          |                              |

          ↓                              ↓


Commercial Success              Success Percentile
Classifier                      Regressor


(XGBoost Classifier)             (Regression Model)

          |                              |

          ↓                              ↓


Success Tier                  Hit Potential Score


0 / 1 / 2 / 3                     0 - 100
```

---

# 10. Final Decision

## Keep:

### Commercial Success Prediction

Use:

```
Tuned XGBoost Classifier
```

Reason:

- Higher Macro F1
- Better discrete tier prediction
- Better balanced classification

---

### Hit Potential Score

Use:

```
Success Percentile Regressor
```

Reason:

- Provides continuous ranking
- Useful for product experience
- Enables more detailed predictions

Example final output:

```
Commercial Prediction:
Successful

Hit Potential:
83 / 100

Market Percentile:
Top 17%
```

---

# Conclusion

The ordinal regression experiment showed that Steam success contains learnable continuous patterns.

However, converting regression outputs into hard classes reduced performance.

Therefore:

- Classification remains the best approach for success tier prediction.
- Regression becomes a complementary scoring model.

Together they create a stronger and more informative prediction system.

```

```

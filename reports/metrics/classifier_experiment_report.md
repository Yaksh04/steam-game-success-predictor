# GameScope — Commercial Success Classifier Experiment Report

## Objective

The goal of the commercial success classifier is to predict a Steam game's market performance tier using only information that would realistically be available around launch time.

The model predicts four classes:

| Class | Meaning       |
| ----- | ------------- |
| 0     | Limited Reach |
| 1     | Niche Success |
| 2     | Successful    |
| 3     | Breakout Hit  |

The main evaluation metric is **Macro F1 Score** instead of accuracy because Steam success distribution is naturally imbalanced.

A model should perform well across all success tiers, not only predict the majority class.

---

# 1. Target Creation

## Problem

Steam ownership follows a long-tail distribution:

- Most games have very few owners
- A small number become extremely successful

Using fixed thresholds would create unrealistic labels.

Example:

```
<10k owners = failure
>1M owners = success
```

This ignores the natural skew of the Steam marketplace.

---

## Solution

Created percentile-based success tiers from SteamSpy estimated owners.

Steps:

1. Converted owner ranges into midpoint values

Example:

```
"200000 - 500000"

↓

350000 owners
```

2. Assigned success tiers based on distribution.

Final target distribution:

| Tier          |  Count |
| ------------- | -----: |
| Limited Reach | 23,502 |
| Niche Success |  8,177 |
| Successful    |  7,630 |
| Breakout Hit  |  5,191 |

This preserves the real-world long-tail nature while keeping enough samples for learning.

---

# 2. Leakage Prevention

A strict leakage audit was performed.

## Removed Post-launch Features

The following features were excluded:

| Feature          | Reason                       |
| ---------------- | ---------------------------- |
| Positive reviews | Available only after release |
| Negative reviews | Available only after release |
| Review count     | Direct popularity signal     |
| Owners           | Used to create target        |
| Player count     | Post-launch popularity       |
| Playtime         | Requires released game       |

The model only uses features realistically available before or during launch.

---

# 3. Baseline Model — Metadata Features Only

## Features Used

Initial launch-time features:

### Price

- price
- is_free
- price category

### Platform Support

- Windows
- Mac
- Linux
- platform count

### Release Features

- release year
- release month
- game age

### Localization

- language count
- log language count

### Achievements

- achievement count
- log achievement count
- achievement availability

### Developer Reputation

Leakage-safe historical features:

- developer previous games
- developer previous success
- developer previous reception
- new developer flag

Feature count:

```
22 features
```

---

## Models Tested

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---

## Results

| Model               | Accuracy | Macro F1 |
| ------------------- | -------: | -------: |
| Logistic Regression |    0.511 |    0.428 |
| Random Forest       |    0.544 |    0.470 |
| XGBoost             |    0.537 |    0.472 |

---

## Observations

The baseline learned meaningful patterns but lacked information about actual game type.

Problem:

Two games:

```
Action RPG

Puzzle Casual Game
```

could appear similar because only genre/tag counts existed.

The model knew:

```
genre_count = 3
```

but not:

```
which genres existed
```

---

# 4. Improvement 1 — Genre Multi-Hot Encoding

## Motivation

Genres are strong market indicators.

Instead of:

```
genre_count = 3
```

Created:

```
genre_Action
genre_RPG
genre_Strategy
genre_Adventure
...
```

using multi-label encoding.

---

## Feature Count

Before:

```
22 features
```

After:

```
50 features
```

---

## Result

Best model: XGBoost

| Version          | Macro F1 |
| ---------------- | -------: |
| Metadata only    |    0.472 |
| + Genre encoding |    0.483 |

Improvement:

```
+0.011 Macro F1
```

---

## Observation

Genres improved class separation, especially between niche and successful games.

However, genres are broad categories and cannot capture detailed market positioning.

---

# 5. Improvement 2 — Steam Tag Encoding

## Motivation

Steam tags contain richer information.

Example:

Both games:

```
Genre:
Action
```

But:

Game A:

```
Open World
RPG
Story Rich
Fantasy
```

Game B:

```
Arcade
Retro
2D
Pixel Graphics
```

have very different markets.

---

## Implementation

Encoded the top 50 most common Steam tags.

Examples:

```
tag_Open_World

tag_Multiplayer

tag_Survival

tag_RPG
```

Rare tags were ignored to avoid sparse noisy features.

---

## Feature Count

Before:

```
50 features
```

After:

```
100 features
```

---

## Results

| Version       | Macro F1 |
| ------------- | -------: |
| Metadata only |    0.472 |
| + Genres      |    0.483 |
| + Tags        |    0.507 |

Improvement:

```
+0.024 Macro F1
```

---

## Observation

Tags provided the largest improvement because they capture game positioning and audience expectations.

---

# 6. Improvement 3 — Publisher Reputation Features

## Motivation

Publisher reputation affects reach.

Example:

Two new developers:

Game A:

```
Developer history:
0 games

Publisher:
Established publisher
```

Game B:

```
Developer history:
0 games

Publisher:
Self published
```

have different success probabilities.

---

## Leakage-Safe Implementation

For each game:

Only publisher history before that game's release date was used.

Created:

- publisher previous games
- publisher previous success
- publisher previous reception
- new publisher flag

Future publisher performance was never used.

---

## Feature Count

Before:

```
100 features
```

After:

```
106 features
```

---

## Results

| Version             | Macro F1 |
| ------------------- | -------: |
| Metadata baseline   |    0.472 |
| + Genres            |    0.483 |
| + Tags              |    0.507 |
| + Publisher history |    0.516 |

---

# Final Pre-Tuning Model Comparison

## Logistic Regression

Accuracy:

```
0.552
```

Macro F1:

```
0.485
```

---

## Random Forest

Accuracy:

```
0.582
```

Macro F1:

```
0.501
```

---

## XGBoost

Accuracy:

```
0.576
```

Macro F1:

```
0.516
```

Selected final model:

```
XGBoost
```

Reasons:

- Best Macro F1
- Handles nonlinear relationships
- Strong performance on tabular data
- Works well with SHAP explainability

---

# Final XGBoost Performance

## Class Performance

| Class         | Precision | Recall |   F1 |
| ------------- | --------: | -----: | ---: |
| Limited Reach |      0.78 |   0.69 | 0.73 |
| Niche Success |      0.30 |   0.34 | 0.32 |
| Successful    |      0.38 |   0.38 | 0.38 |
| Breakout Hit  |      0.57 |   0.72 | 0.64 |

---

# Overfitting Analysis

Training Macro F1:

```
0.683
```

Testing Macro F1:

```
0.516
```

Conclusion:

Some expected tree-model overfitting exists, but the gap is acceptable.

Further tuning should focus on regularization.

---

# Final Feature Engineering Impact

Overall improvement:

```
Macro F1:

0.472
    ↓
0.516
```

Relative improvement:

≈9%

All improvements were achieved using valid launch-time features without introducing data leakage.

---

# Next Step

Perform hyperparameter optimization using RandomizedSearchCV.

Optimization target:

```
Macro F1 Score
```

Focus:

- Reduce overfitting
- Improve middle tier classification
- Maintain breakout hit detection performance

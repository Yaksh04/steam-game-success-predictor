# GameScope — Steam Game Success Prediction Platform

# Complete Project Development Report

## Project Objective

GameScope is an end-to-end machine learning system designed to predict the potential success of upcoming Steam games using information realistically available before or around launch.

The project aims to answer three questions:

1. How commercially successful will a game become?
2. How well will players receive the game?
3. What is the overall hit potential of the game?

Final planned outputs:

- Commercial Success Tier Prediction
- Player Reception Prediction
- Overall Hit Potential Score
- Model Explanations
- Interactive Streamlit Application

---

# 1. Dataset Collection

## Data Source

Dataset:

Steam games dataset collected from Kaggle.

Original dataset:

- ~120,000 Steam games
- JSON format
- 34 raw attributes

Raw file:

```text
data/raw/games.json
```

The dataset contains:

- game metadata
- price information
- supported platforms
- genres
- tags
- languages
- developer information
- publisher information
- Steam ownership estimates
- review statistics

---

# 2. Initial Data Processing

## Data Loading Pipeline

Implemented:

```text
src/data/load_data.py
```

Responsibilities:

- Load JSON data
- Convert nested structures into usable tabular format
- Validate schema

Generated:

```text
data/interim/steam_games_schema.csv
```

---

## Data Cleaning Pipeline

Implemented:

```text
src/data/clean_data.py
```

Cleaning steps:

- Removed invalid entries
- Handled missing values
- Parsed release dates
- Converted data types
- Removed unusable columns

Output:

```text
data/interim/steam_games_clean.csv
```

---

# 3. Dataset Filtering Decisions

The original dataset contained many older games and incomplete upcoming releases.

Decision:

Use games released:

```text
2020 - 2025
```

Reason:

- modern Steam market behavior
- enough historical data
- avoids incomplete 2026 entries

Final dataset:

```text
44,500 games
```

---

# 4. Exploratory Data Analysis

Notebook:

```text
notebooks/02_eda.ipynb
```

Reports:

```text
reports/metrics/eda_summary.md
```

Visualizations:

```text
reports/figures/
```

---

## Main Findings

### Steam Market Distribution

Finding:

Steam success follows a strong long-tail distribution.

Most games:

- low ownership
- limited visibility

Small percentage:

- very large audience

Conclusion:

Fixed owner thresholds would create poor targets.

---

## Pricing Analysis

Observation:

Higher performing games generally have higher average prices.

Interpretation:

Price represents:

- production quality
- positioning
- market expectations

---

## Platform Analysis

Observation:

Games supporting multiple platforms generally achieve higher reach.

Features created:

- Windows support
- Mac support
- Linux support
- platform count

---

## Release Analysis

Steam releases increased heavily over time.

Problem:

Older games have had more time to collect owners.

Solution:

Create time-aware features.

---

# 5. Target Engineering

Implemented:

```text
src/features/create_targets.py
```

---

# Commercial Success Target

Problem:

Raw owners are extremely skewed.

Example:

A few games:

Millions of owners

Most games:

Very few owners

---

## Solution: Percentile-Based Success Classes

Process:

SteamSpy range:

```text
200000 - 500000
```

converted into:

```text
owners_midpoint = 350000
```

Then percentile ranking was calculated.

Final classes:

| Class | Meaning       |
| ----- | ------------- |
| 0     | Limited Reach |
| 1     | Niche Success |
| 2     | Successful    |
| 3     | Breakout Hit  |

Final distribution:

| Class         | Samples |
| ------------- | ------: |
| Limited Reach |  23,502 |
| Niche Success |   8,177 |
| Successful    |   7,630 |
| Breakout Hit  |   5,191 |

---

# 6. Leakage Prevention

Major focus of the project.

Removed from model inputs:

| Feature          | Reason                |
| ---------------- | --------------------- |
| estimated owners | used to create target |
| owner midpoint   | direct popularity     |
| positive reviews | post launch           |
| negative reviews | post launch           |
| review counts    | future information    |
| player count     | post launch           |
| playtime         | post launch           |

Goal:

Only use information available before launch.

---

# 7. Feature Engineering

Implemented:

```text
src/features/build_features.py
```

Output:

```text
data/processed/steam_games_features.csv
```

---

# Initial Feature Set

Started with:

22 metadata features.

Categories:

## Price

- price
- is_free
- price_category

## Platform

- windows
- mac
- linux
- platform_count

## Release

- release_year
- release_month
- game_age

## Localization

- language_count
- log_languages

## Achievements

- achievement_count
- log_achievements
- has_achievements

---

# Developer Reputation Features

Created historical features:

- developer_previous_games
- developer_previous_success
- developer_previous_reception
- new_developer

Important:

Only games released before the current game were considered.

This prevents future leakage.

---

# Genre Encoding Experiment

Problem:

Only genre count loses information.

Example:

Both:

```
genre_count = 3
```

but:

Game A:

RPG, Adventure, Open World

Game B:

Puzzle, Casual, Family

are very different.

Solution:

Multi-hot encode genres.

Feature count:

```
22 → 50
```

---

# Steam Tag Encoding Experiment

Tags contain detailed positioning.

Examples:

- Open World
- Story Rich
- Multiplayer
- Survival

Encoded:

Top 50 Steam tags.

Feature count:

```
50 → 100
```

---

# Publisher Reputation Features

Motivation:

Publisher influence affects reach.

Added:

- publisher_previous_games
- publisher_previous_success
- publisher_previous_reception
- new_publisher

Calculated historically only.

Feature count:

```
100 → 106+
```

---

# 8. Success Classification Experiments

Evaluation metric:

Macro F1 Score

Reason:

Success classes are imbalanced.

Accuracy alone would favor the majority class.

---

# Baseline Results

22 metadata features:

| Model               | Macro F1 |
| ------------------- | -------: |
| Logistic Regression |    0.428 |
| Random Forest       |    0.470 |
| XGBoost             |    0.472 |

---

# Feature Improvement Timeline

## Genre Encoding

Result:

```text
0.472 → 0.483
```

Improvement:

+0.011

---

## Steam Tags

Result:

```text
0.483 → 0.507
```

Improvement:

+0.024

---

## Publisher Reputation

Result:

```text
0.507 → 0.516
```

---

# Hyperparameter Optimization

Used:

RandomizedSearchCV

Settings:

- 30 iterations
- 3-fold cross validation
- scoring = Macro F1

Best model:

XGBoost

Best parameters:

```python
n_estimators = 457
max_depth = 7
learning_rate = 0.075
subsample = 0.916
colsample_bytree = 0.743
```

---

# Final Success Classifier

Performance:

```text
Accuracy: 0.600

Macro F1: 0.532
```

Saved:

```text
models/classifier/final_xgboost.pkl
```

---

# 9. Ordinal Success Regression Experiment

Question:

Can success prediction improve by treating success as continuous?

---

## Target

Instead of:

```text
success_tier
```

predict:

```text
owner_percentile
```

---

# Results

Regression:

| Metric | Value |
| ------ | ----: |
| MAE    | 0.162 |
| R²     | 0.422 |

The model learned meaningful market ranking.

---

Converted back into classes:

```text
Accuracy: 0.50
Macro F1: 0.442
```

Comparison:

| Model              | Macro F1 |
| ------------------ | -------: |
| XGBoost Classifier |    0.532 |
| Ordinal Regression |    0.442 |

---

# Decision

Do not replace classifier.

Use both:

```
Classifier
    ↓
Success Tier


Regressor
    ↓
Hit Potential Score
```

This provides richer predictions.

---

# Current Final Architecture

Input Game Data

        |

Feature Engineering

        |

---

| |

Success Classifier Success Regressor

| |

Tier Prediction Player reception Predicton

---

# Current Project Status

Completed:

✅ Data loading pipeline

✅ Cleaning pipeline

✅ EDA

✅ Target engineering

✅ Leakage audit

✅ Feature engineering

✅ Baseline models

✅ XGBoost optimization

✅ Success regression experiment

✅ Model comparison reports

---

# Remaining Work

## 1. Player Reception Prediction

Build:

```text
reception_score model
```

Predict expected user satisfaction.

---

## 2. Probability Calibration

Improve confidence reliability.

Methods:

- calibration curves
- Brier score
- CalibratedClassifierCV

---

## 3. Explainability

Add SHAP.

Global:

Which features influence Steam success overall?

Local:

Why did this game receive this prediction?

---

## 4. Prediction Engine

Combine:

- success classifier
- success regressor
- reception model

Generate:

Overall Hit Score.

---

## 5. Streamlit Application

Build:

- Steam lookup
- prediction dashboard
- SHAP explanations
- analytics

---

## 6. Deployment

Final steps:

- clean README
- deployment
- example predictions
- documentation

---

# Current Evaluation

The project successfully evolved from a basic ML classifier into a complete predictive analytics system.

Key strengths:

- realistic prediction constraints
- no target leakage
- temporal feature engineering
- multiple experiments
- documented decisions
- interpretable ML direction

Current state:

Production pipeline development phase.

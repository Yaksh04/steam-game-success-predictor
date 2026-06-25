"""
Market Reach Classification Training Pipeline

Predicts:
Limited Reach
emerging Performer
market success
High Impact

Uses only launch-time features.
"""

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.utils.class_weight import compute_sample_weight

import joblib

DATA_PATH = Path("data/processed/steam_games_features.csv")
MODEL_PATH = Path("models/classifier")
REPORT_PATH = Path("reports/metrics")


# Loading the data
def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)
    return df


# Define the features, remove unwanted columns for now, starting simpler
def split_features_target(df):

    remove_columns = [
        # targets
        "success_tier",
        "reception_score",
        # leakage
        "owners_midpoint",
        "owner_percentile",
        "estimated_owners",
        "positive",
        "negative",
        "total_reviews",
        # identifiers
        "appid",
        "name",
        # raw columns already engineered
        "release_date",
        "developers",
        "publishers",
        "main_developer",
        "genres",
        "categories",
        "tags",
        "supported_languages",
        "achievements",
    ]

    X = df.drop(columns=[col for col in remove_columns if col in df.columns])

    y = df["success_tier"]

    return X, y


# sepererate preprocessor function to make model tuning easy
def build_preprocessor(X):
    numeric_features = [
        "price",
        "platform_count",
        "release_year",
        "release_month",
        "game_age",
        "developer_previous_games",
        "log_developer_previous_games",
        "developer_previous_success",
        "developer_previous_reception",
        "publisher_previous_games",
        "log_publisher_previous_games",
        "publisher_previous_success",
        "publisher_previous_reception",
        "language_count",
        "log_languages",
        "achievement_count",
        "log_achievements",
    ]

    # adding genre features in numeric features after encoding in build_features.py
    genre_features = [col for col in X.columns if col.startswith("genre_")]
    numeric_features += genre_features

    # adding tag features after encoding them in build_features.py
    tag_features = [col for col in X.columns if col.startswith("tag_")]
    numeric_features += tag_features

    # Binary features are also numeric as they are already encoded
    numeric_features += [
        "is_free",
        "windows",
        "mac",
        "linux",
        "new_developer",
        "new_publisher",
        "has_achievements",
    ]
    categorical_features = ["price_category"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "scaler",
                StandardScaler(),
            ),  # standard scaler needed for logistic regression
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_classifier():
    df = load_data()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )  # stratify preserves the class distribution

    print("Training samples:", X_train.shape)

    print("Testing samples:", X_test.shape)

    # refactored the pipeline by building seperate preprocessor function so that same function can be used in hyper-parameter tuning
    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        ),
    }

    # Training and evaluation pipeline
    results = []
    best_model = None
    best_score = 0

    # adding sample weights for XGBoost
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    for name, classifier in models.items():

        print("\nTraining:", name)

        model = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", classifier)]
        )

        if name == "XGBoost":
            model.fit(X_train, y_train, classifier__sample_weight=sample_weights)

        else:
            model.fit(X_train, y_train)

        # Check for overfitting
        train_preds = model.predict(X_train)

        train_f1 = f1_score(y_train, train_preds, average="macro")

        print("Train Macro F1:", train_f1)

        # prediction
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        f1 = f1_score(y_test, preds, average="macro")

        results.append({"model": name, "accuracy": acc, "macro_f1": f1})

        print(classification_report(y_test, preds))

        if f1 > best_score:

            best_score = f1

            best_model = model

            best_predictions = preds

            best_name = name

    results_df = pd.DataFrame(results)

    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_PATH / "classifier_results.csv", index=False)

    print(results_df)
    print("Metrics saved")

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(y_test, best_predictions, ax=ax)

    plt.title(f"{best_name} Confusion Matrix")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    plt.savefig("reports/figures/classifier_confusion_matrix.png", bbox_inches="tight")

    MODEL_PATH.mkdir(exist_ok=True)

    joblib.dump(best_model, MODEL_PATH / "best_classifier.pkl")
    print("Best model saved")


if __name__ == "__main__":

    train_classifier()

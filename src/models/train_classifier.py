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

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import joblib

DATA_PATH = Path("data/processed/steam_games_features.csv")
MODEL_PATH = Path("models/classifier")


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


def train_classifier():

    df = load_data()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )  # stratify preserves the class distribution

    print("Training samples:", X_train.shape)

    print("Testing samples:", X_test.shape)

    numeric_features = [
        "price",
        "genre_count",
        "platform_count",
        "release_year",
        "release_month",
        "game_age",
        "developer_previous_games",
        "log_developer_previous_games",
        "developer_previous_success",
        "developer_previous_reception",
        "language_count",
        "log_languages",
        "tag_count",
        "achievement_count",
        "log_achievements",
    ]
    # Binary features are also numeric as they are already encoded
    numeric_features += [
        "is_free",
        "windows",
        "mac",
        "linux",
        "new_developer",
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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    print("Training Logistic Regression...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(y_test, predictions, average="macro")

    print("Accuracy:", accuracy)

    print("Macro F1:", macro_f1)

    print(classification_report(y_test, predictions))


if __name__ == "__main__":

    train_classifier()

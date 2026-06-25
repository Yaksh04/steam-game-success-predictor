import pandas as pd
import numpy as np

from pathlib import Path

import joblib

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    f1_score,
    classification_report,
)

from xgboost import XGBRegressor

from train_classifier import (
    load_data,
    split_features_target,
    build_preprocessor,
)


def percentile_to_tier(values):

    tiers = []

    for value in values:

        if value < 0.50:

            tiers.append(0)

        elif value < 0.80:

            tiers.append(1)

        elif value < 0.95:

            tiers.append(2)

        else:

            tiers.append(3)

    return np.array(tiers)


# Training function
def train_success_regressor():

    df = load_data()

    X, _ = split_features_target(df)

    y = df["owner_percentile"]

    y_tier = df["success_tier"]

    X_train, X_test, y_train, y_test, tier_train, tier_test = train_test_split(
        X, y, y_tier, test_size=0.2, random_state=42, stratify=y_tier
    )

    model = Pipeline(
        [
            ("preprocessor", build_preprocessor(X)),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training ordinal success model...")

    model.fit(X_train, y_train)

    percentile_preds = model.predict(X_test)

    percentile_preds = np.clip(percentile_preds, 0, 1)

    # Evaluate the model
    print("MAE:", mean_absolute_error(y_test, percentile_preds))

    print("R2:", r2_score(y_test, percentile_preds))

    # Convert back to tiers
    predicted_tiers = percentile_to_tier(percentile_preds)

    print(classification_report(tier_test, predicted_tiers))

    print("Macro F1:", f1_score(tier_test, predicted_tiers, average="macro"))

    # Save the model
    joblib.dump(model, "models/regressor/success_regressor.pkl")


if __name__ == "__main__":

    train_success_regressor()

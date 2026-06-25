import pandas as pd

from pathlib import Path

import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import f1_score, classification_report

from sklearn.utils.class_weight import compute_sample_weight

from scipy.stats import randint, uniform

from train_classifier import load_data, split_features_target

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from train_classifier import build_preprocessor


df = load_data()

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

weights = compute_sample_weight(class_weight="balanced", y=y_train)

pipeline = Pipeline(
    [
        ("preprocessor", build_preprocessor(X)),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss", random_state=42
            ),  # objective and eval_metric set to handle multi-class classification (0,1,2,3)
        ),
    ]
)

# Search space
params = {
    "classifier__n_estimators": randint(200, 600),
    "classifier__max_depth": randint(3, 8),
    "classifier__learning_rate": uniform(0.01, 0.1),
    "classifier__subsample": uniform(0.7, 0.3),
    "classifier__colsample_bytree": uniform(0.7, 0.3),
}

search = RandomizedSearchCV(
    pipeline,
    params,
    n_iter=30,
    scoring="f1_macro",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)


search.fit(X_train, y_train, classifier__sample_weight=weights)

# Evaluation
preds = search.predict(X_test)

print(classification_report(y_test, preds))

print(search.best_params_)

print(f1_score(y_test, preds, average="macro"))

# Save the model
joblib.dump(search.best_estimator_, "models/classifier/final_xgboost.pkl")

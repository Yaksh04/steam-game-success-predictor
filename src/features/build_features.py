"""
Feature Engineering Pipeline

Converts cleaned Steam metadata into
machine learning features.

Only launch-time information is used.
Post-launch metrics are excluded to
avoid data leakage.
"""

import pandas as pd
import numpy as np
import ast

from pathlib import Path


INPUT_PATH = Path("data/processed/steam_games_targets.csv")
OUTPUT_PATH = Path("data/processed/steam_games_features.csv")


# helper function for list columns
def parse_list(value):

    if isinstance(value, list):
        return value

    try:
        return ast.literal_eval(value)

    except:
        return []


# Creating genre features
def create_genre_features(df):

    df["genres"] = df["genres"].apply(parse_list)

    df["genre_count"] = df["genres"].apply(len)

    return df


# Creating price features
def create_price_features(df):

    df["is_free"] = (df["price"] == 0).astype(int)

    def categorize_price(price):

        if price == 0:

            return "Free"

        elif price < 10:

            return "Budget"

        elif price < 30:

            return "Standard"

        else:

            return "Premium"

    df["price_category"] = df["price"].apply(categorize_price)

    return df


# Creating platform-count feature
def create_platform_features(df):

    platform_columns = ["windows", "mac", "linux"]

    for col in platform_columns:

        df[col] = df[col].astype(int)

    df["platform_count"] = df[platform_columns].sum(axis=1)

    return df


# Creat release feature
def create_release_features(df):

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    df["release_year"] = df["release_date"].dt.year

    df["release_month"] = df["release_date"].dt.month

    df["game_age"] = 2025 - df["release_year"]

    return df


# Create language support feature
def create_language_features(df):

    df["supported_languages"] = df["supported_languages"].apply(parse_list)

    df["language_count"] = df["supported_languages"].apply(len)

    # We cap language count at 30 as some games listed 103 languages which has low probability
    df["language_count"] = df["language_count"].clip(upper=30)

    df["log_languages"] = np.log1p(df["language_count"])

    return df


# Create tag feature which represents market positioning
def create_tag_features(df):

    df["tags"] = df["tags"].apply(parse_list)

    df["tag_count"] = df["tags"].apply(len)

    return df


# Create achievements feature, succesfull game has higher changes of having >0 achievements
def create_achievement_features(df):

    df["achievement_count"] = df["achievements"].fillna(0)

    # achievements capped at 200 to avoid exploitation
    df["achievement_count"] = df["achievement_count"].clip(upper=200)

    df["has_achievements"] = (df["achievement_count"] > 0).astype(int)

    # Creating log achievements as well for linear models, since we will compare different models
    df["log_achievements"] = np.log1p(df["achievement_count"])

    return df


# Now we create Developers feature, this is probably the most important and leakage prone
def create_developer_features(df):

    df["developers"] = df["developers"].apply(parse_list)

    df["main_developer"] = df["developers"].apply(
        lambda x: x[0] if len(x) > 0 else "Unknown"
    )

    df = df.sort_values("release_date")

    df["developer_previous_games"] = df.groupby("main_developer").cumcount()

    # again, doing log of previous games number to diminish the difference, useful for linear models
    df["log_developer_previous_games"] = np.log1p(df["developer_previous_games"])

    # developers previous success
    df["developer_previous_success"] = df.groupby("main_developer")[
        "success_tier"
    ].transform(
        lambda x: x.shift().expanding().mean()
    )  # using shift to ignore current game

    # Devs previous reception
    df["developer_previous_reception"] = df.groupby("main_developer")[
        "reception_score"
    ].transform(lambda x: x.shift().expanding().mean())

    # new developer will have NaN because no previous games
    df["new_developer"] = (df["developer_previous_games"] == 0).astype(int)

    df[["developer_previous_success", "developer_previous_reception"]] = df[
        ["developer_previous_success", "developer_previous_reception"]
    ].fillna(0)

    return df


# full feature engineering pipeline
def build_features():

    df = pd.read_csv(INPUT_PATH)

    print("Loaded:", df.shape)

    df = create_genre_features(df)

    df = create_price_features(df)

    df = create_platform_features(df)

    df = create_release_features(df)

    df = create_developer_features(df)

    df = create_language_features(df)

    df = create_tag_features(df)

    df = create_achievement_features(df)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)

    print("Final shape:", df.shape)


if __name__ == "__main__":

    build_features()

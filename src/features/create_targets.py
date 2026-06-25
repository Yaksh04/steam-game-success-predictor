"""
Target Engineering

Creates:

1. Market Reach Tier

Predicts commercial adoption using Steam owner estimates.

Classes:
0 - Limited Reach
1 - Emerging Performer
2 - Market Success
3 - High impact


2. Reception Score

Predicts expected player satisfaction.

Formula:
positive_reviews /
(total_reviews)


Important:

Post-launch signals are removed before training
to prevent data leakage.
"""

import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/interim/steam_games_clean.csv")

OUTPUT_PATH = Path("data/processed/steam_games_targets.csv")


# extract estimated owners from owner range
def convert_owners(owner_range):

    try:
        low, high = owner_range.split("-")

        low = int(low.strip())
        high = int(high.strip())

        return (low + high) / 2

    except:
        return None


# I have converted ownership estimates to percentile based market reach tiers to create a more stable and business-interpretable classification problem


# so basically current success tier measure if a game will perform well in steam ecosystem compared to other games using percentile
def create_success_target(df):

    df["owners_midpoint"] = df["estimated_owners"].apply(convert_owners)

    df = df.dropna(subset=["owners_midpoint"])

    df["owner_percentile"] = df["owners_midpoint"].rank(pct=True, method="average")

    def assign_tier(percentile):

        if percentile < 0.50:
            return 0

        elif percentile < 0.80:
            return 1

        elif percentile < 0.95:
            return 2

        else:
            return 3

    df["success_tier"] = df["owner_percentile"].apply(assign_tier)

    return df


def create_reception_target(df):
    df["total_reviews"] = df["positive"] + df["negative"]

    # review threshold is kept 20 as steam also doesnt consider tiny review counts meaningful
    df = df[df["total_reviews"] >= 20]

    df["reception_score"] = df["positive"] / df["total_reviews"]
    return df


# function to remove post launch features
def remove_leakage_columns(df):

    leakage_columns = ["estimated_owners", "positive", "negative"]

    df = df.drop(columns=leakage_columns)

    return df


def create_targets():

    df = pd.read_csv(INPUT_PATH)

    print("Loaded: ", df.shape)

    df = create_success_target(df)
    df = create_reception_target(df)
    df = remove_leakage_columns(df)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved: ", OUTPUT_PATH)

    print(df["success_tier"].value_counts())


if __name__ == "__main__":
    create_targets()

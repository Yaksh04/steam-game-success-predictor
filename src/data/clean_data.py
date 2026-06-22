import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/interim/steam_games_schema.csv")

OUTPUT_PATH = Path("data/interim/steam_games_clean.csv")


# Function to load the data
def load_data():

    df = pd.read_csv(INPUT_PATH)

    print("Loaded:", df.shape)

    return df


# Function to select useful columns
def select_columns(df):

    columns = [
        # identifiers
        "appid",
        "name",
        # release info
        "release_date",
        # companies
        "developers",
        "publishers",
        # game properties
        "price",
        "genres",
        "categories",
        "tags",
        "supported_languages",
        "achievements",
        # platforms
        "windows",
        "mac",
        "linux",
        # target creation ONLY
        "estimated_owners",
        "positive",
        "negative",
    ]

    return df[columns]


# function to clean messy dates
def clean_dates(df):
    df["release_date"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    )  # convert to datetime, and turn messy dates into NaT

    df = df.dropna(subset=["release_date"])  # drop rows with NaT

    return df


def filter_release_year(df):

    # remove games released after 2025 because many of the 2026 games data is incomplete
    df = df[df["release_date"].dt.year <= 2025]

    return df


# fucntion to fix numeric columns
def fix_numeric(df):

    numeric_cols = ["price", "positive", "negative", "achievements"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


# function to remove impossible games
def remove_invalid_rows(df):

    df = df.dropna(subset=["estimated_owners"])

    # remove games with no reviews as they will not help
    df = df[df["positive"] + df["negative"] > 0]

    return df


# Making a remove duplicates function because 186 duplicates were found during feature engineering
def remove_duplicates(df):

    before = len(df)

    # name alone can be dangerous, so name + date
    df = df.drop_duplicates(subset=["name", "release_date"], keep="first")

    after = len(df)

    print(f"Removed duplicates: {before-after}")

    return df


def clean_pipeline():

    df = load_data()
    df = select_columns(df)
    df = clean_dates(df)
    df = filter_release_year(df)
    df = remove_duplicates(df)
    df = fix_numeric(df)
    df = remove_invalid_rows(df)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    clean_pipeline()

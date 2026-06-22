"""
Target Engineering

Creates:

1. Market Reach Tier

Predicts commercial adoption using Steam owner estimates.

Classes:
0 - Limited Reach
1 - Emerging Performer
2 - Market Success
3 - Breakout Hit


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

INPUT_PATH = Path(
    "data/interim/steam_games_clean.csv"
)

OUTPUT_PATH = Path(
    "data/processed/steam_games_targets.csv"
)

# extract estimated owners from owner range
def convert_owners(owner_range):

  try :
     low,high = owner_range.split("-")

     low= int(low.strip())
     high = int(high.strip())

     return (low+high) / 2;
  
  except : 
    return None 


def create_success_target(df):


    df["owners_midpoint"] = (
        df["estimated_owners"]
        .apply(convert_owners)
    )


    df = df.dropna(
        subset=["owners_midpoint"]
    )


    def assign_success_tier(owners):


        if owners < 50000:

            return 0


        elif owners < 200000:

            return 1


        elif owners < 1000000:

            return 2


        else:

            return 3



    df["success_tier"] = (
        df["owners_midpoint"]
        .apply(assign_success_tier)
    )


    return df



def create_reception_target(df): 
  df["reception_score"] = df['positive']/(df['positive'] + df['negative'])

  return df


# function to remove post launch features
def remove_leakage_columns(df):

  leakage_columns = [
    'estimated_owners',
    'positive',
    'negative'
  ]

  df=df.drop(
    columns=leakage_columns
  )

  return df


def create_targets():

  df=pd.read_csv(INPUT_PATH)

  print('Loaded: ', df.shape)

  df=create_success_target(df)
  df=create_reception_target(df)
  df=remove_leakage_columns(df)

  OUTPUT_PATH.parent.mkdir(exist_ok=True)

  df.to_csv(OUTPUT_PATH, index=False)

  print('Saved: ', OUTPUT_PATH)

  print(
    df[
      'success_tier'
    ].value_counts()
  )

if __name__=='__main__':
  create_targets()
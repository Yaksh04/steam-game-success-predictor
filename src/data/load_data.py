import json

from pathlib import Path

import pandas as pd


from src.data.validation import (
    validate_columns,
    check_duplicates
)




RAW_PATH = (
    "data/raw/games.json"
)


SAVE_PATH = (
    "data/interim/steam_games_schema.csv"
)



def load_json_data(path):


    with open(
        path,
        encoding="utf-8"
    ) as f:


        data=json.load(f)



    df=pd.DataFrame.from_dict(
        data,
        orient="index"
    )


    df.reset_index(
        inplace=True
    )


    df.rename(
        columns={
            "index":"appid"
        },
        inplace=True
    )


    return df




def save_interim_data():


    df=load_json_data(
        RAW_PATH
    )


    print(
        "Loaded:",
        df.shape
    )


    validate_columns(df)

    check_duplicates(df)



    df.to_csv(
        SAVE_PATH,
        index=False
    )


    print(
        "Saved:",
        SAVE_PATH
    )





if __name__=="__main__":


    save_interim_data()
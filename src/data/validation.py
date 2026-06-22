REQUIRED_COLUMNS = [


    "name",

    "release_date",

    "developers",

    "publishers",

    "price",

    "genres",

    "positive",

    "negative",

    "estimated_owners"

]




def validate_columns(df):


    missing=[]


    for column in REQUIRED_COLUMNS:


        if column not in df.columns:

            missing.append(column)



    if missing:

        raise Exception(
            f"Missing columns: {missing}"
        )


    print(
        "Column validation passed"
    )



def check_duplicates(df):


    duplicate_count = (
        df["appid"]
        .duplicated()
        .sum()
    )


    print(
        "Duplicate appids:",
        duplicate_count
    )
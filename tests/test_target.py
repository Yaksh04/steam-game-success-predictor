import pandas as pd


from src.features.create_targets import (
    create_reception_target
)



def test_reception_score():


    df=pd.DataFrame({

        "positive":[80],

        "negative":[20]

    })


    result=create_reception_target(df)


    assert (
        result["reception_score"]
        .iloc[0]
        ==
        0.8
    )
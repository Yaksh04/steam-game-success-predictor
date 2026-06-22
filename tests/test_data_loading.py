from src.data.load_data import load_json_data



def test_json_loading():


    df=load_json_data(
        "data/raw/games.json"
    )


    assert df.shape[0] > 100000


    assert "appid" in df.columns
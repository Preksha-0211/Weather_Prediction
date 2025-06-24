import pandas as pd
from features import create_features

def test_create_features_columns_exist():
    df = pd.DataFrame({"temp": [1, 2]}, index=pd.to_datetime(["2021-01-01 00:00", "2021-01-01 01:00"]))
    result = create_features(df)
    for col in ["hour", "month", "year", "dayofmonth"]:
        assert col in result.columns


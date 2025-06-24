import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time series features based on the DataFrame index."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofmonth'] = df.index.day
    return df

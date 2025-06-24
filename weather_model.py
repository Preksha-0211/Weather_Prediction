import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


TARGET_COL = "contest-tmp2m-14d__tmp2m"
CATEGORY_COL = "climateregions__climateregion"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofmonth'] = df.index.day
    return df


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill()


def encode_categorical(train: pd.DataFrame, test: pd.DataFrame, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
        train[CATEGORY_COL] = encoder.fit_transform(train[CATEGORY_COL])
    else:
        train[CATEGORY_COL] = encoder.transform(train[CATEGORY_COL])
    test[CATEGORY_COL] = encoder.transform(test[CATEGORY_COL])
    return train, test, encoder


def handle_outliers(df: pd.DataFrame, columns):
    for col in columns:
        feat = df[col]
        upper_limit = feat.mean() + 3 * feat.std()
        lower_limit = feat.mean() - 3 * feat.std()
        df[col] = np.where(
            feat > upper_limit,
            upper_limit,
            np.where(feat < lower_limit, lower_limit, feat),
        )
    return df


def preprocess(train: pd.DataFrame, test: pd.DataFrame, encoder=None):
    train = create_features(train)
    test = create_features(test)
    train = fill_na(train)
    test = fill_na(test)
    train, test, encoder = encode_categorical(train, test, encoder)
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    train = handle_outliers(train, numeric_cols)
    test = handle_outliers(test, numeric_cols.intersection(test.columns))
    return train, test, encoder


def train_model(train_path: str, test_path: str, model_out: str):
    train = pd.read_csv(train_path, parse_dates=['startdate'])
    test = pd.read_csv(test_path, parse_dates=['startdate'])
    train.set_index('startdate', inplace=True)
    test.set_index('startdate', inplace=True)

    train, test, encoder = preprocess(train, test)

    X = train.drop(TARGET_COL, axis=1)
    y = train[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    reg = XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        n_estimators=80,
        early_stopping_rounds=5,
        objective='reg:squarederror',
        max_depth=5,
        learning_rate=0.02,
    )

    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=0)

    joblib.dump({'model': reg, 'encoder': encoder}, model_out)
    return reg


def predict(model_in: str, test_path: str, output_csv: str):
    bundle = joblib.load(model_in)
    reg = bundle['model']
    encoder = bundle['encoder']

    test = pd.read_csv(test_path, parse_dates=['startdate'])
    test.set_index('startdate', inplace=True)

    _, test, _ = preprocess(test, test, encoder)
    preds = reg.predict(test)

    pd.DataFrame({TARGET_COL: preds, 'index': test.index}).to_csv(output_csv, index=False)
    return preds

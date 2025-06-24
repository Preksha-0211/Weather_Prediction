import pandas as pd


def load_data(path="submission.csv"):
    """Load predictions file and return DataFrame."""
    return pd.read_csv(path)


def predict():
    df = load_data()
    mean_value = df[df.columns[0]].mean()
    print(f"Mean prediction: {mean_value}")


if __name__ == "__main__":
    predict()

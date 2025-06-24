import argparse
from weather_model import train_model, predict


def main():
    parser = argparse.ArgumentParser(description="Weather prediction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument("train_csv", help="Path to training CSV")
    train_p.add_argument("test_csv", help="Path to test CSV")
    train_p.add_argument("model_out", help="Where to save the trained model")

    pred_p = subparsers.add_parser("predict", help="Generate predictions")
    pred_p.add_argument("model_in", help="Path to saved model")
    pred_p.add_argument("test_csv", help="Path to test CSV")
    pred_p.add_argument("output_csv", help="Where to save predictions")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.train_csv, args.test_csv, args.model_out)
    elif args.command == "predict":
        predict(args.model_in, args.test_csv, args.output_csv)


if __name__ == "__main__":
    main()

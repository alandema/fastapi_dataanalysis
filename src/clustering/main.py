from dotenv import load_dotenv  # noqa
load_dotenv('config.env')  # noqa

import argparse
import pandas as pd
from helpers.process_data import process_data


def main(action, csv_path):
    if action == "train":
        train_data = pd.read_csv(csv_path)

        # Add your training logic here
        print(f"Training with data from {csv_path}")
    elif action == "test":
        # Add your testing logic here
        print(f"Testing with data from {csv_path}")
    else:
        print("Invalid action. Use 'train' or 'test'.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run clustering operations")
    parser.add_argument("action", choices=["train", "test"], help="Action to perform")
    parser.add_argument("csv_path", help="Path to the CSV file")

    args = parser.parse_args()
    main(args.action, args.csv_path)

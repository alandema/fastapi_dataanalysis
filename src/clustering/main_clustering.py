from dotenv import load_dotenv  # noqa
load_dotenv('config.env')  # noqa

import argparse
import pandas as pd
from helpers.pre_process_data import deal_with_missing_data, encode_categorical_columns, scale_data
from helpers.clustering import cluster_dataframe
from helpers.metrics import get_metrics


def main(action, csv_path, label_column_index=None):
    if action == "fit":
        print(f"Fit with data from {csv_path}")
        feature_data, label_encoders, label_column = preprocess_daata(csv_path, label_column_index)
        cluster_labels = cluster_data(feature_data)
        evaluate_data(label_column, cluster_labels, feature_data)
    else:
        print("Invalid action. Use 'fit'.")


def preprocess_daata(csv_path, label_column_index):
    data = pd.read_csv(csv_path)
    data = deal_with_missing_data(data)

    label_column = data.iloc[:, label_column_index]
    feature_data = data.drop(data.columns[label_column_index], axis=1)

    feature_data, label_encoders = encode_categorical_columns(feature_data)
    feature_data = scale_data(feature_data)

    return feature_data, label_encoders, label_column


def cluster_data(feature_data):
    clustered_data = cluster_dataframe(feature_data)
    return clustered_data


def evaluate_data(label_column, cluster_labels, feature_data):
    metrics = get_metrics(label_column, cluster_labels, feature_data)
    print(metrics)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="Run clustering operations")
    # parser.add_argument("action", choices=["fit"], help="Action to perform")
    # parser.add_argument("csv_path", help="Path to the CSV file")
    # parser.add_argument("label_column_index", help="Path to the CSV file")

    # args = parser.parse_args()
    # main(args.action, args.csv_path)
    main('fit', 'src/clustering/data/obesity.csv', -1)

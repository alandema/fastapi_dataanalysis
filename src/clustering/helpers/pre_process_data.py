
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.preprocessing import StandardScaler


def deal_with_missing_data(df):
    """
    Remove rows with missing data from the given DataFrame.

    Args:
    df (pandas.DataFrame): The input DataFrame

    Returns:
    pandas.DataFrame: A new DataFrame with rows containing missing data removed
    """
    return df.dropna(axis=0)


def encode_categorical_columns(df):
    prepared_df = df.copy()
    label_encoders = {}

    for column in prepared_df.columns:
        try:
            # Try to convert the column to numeric
            prepared_df[column] = pd.to_numeric(prepared_df[column])
        except ValueError:
            # If conversion fails, use label encoding
            le = LabelEncoder()
            prepared_df[column] = le.fit_transform(prepared_df[column].astype(str))
            label_encoders[column] = le

    return prepared_df, label_encoders


def scale_data(df):
    # Create a StandardScaler object
    scaler = StandardScaler()

    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df

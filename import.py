#!/usr/bin/env python3
import pandas as pd
import pickle
import sys



def load_data(file_path):
    df = pd.read_csv(file_path)

    # Print first 5 rows
    print("\nFirst 5 rows of the dataset:\n")
    print(df.head(), "\n")
    print("-" * 120, "\n")  # Divider

    # Print dataset information
    print("Dataset Information:\n")
    print(df.info(), "\n")
    print("-" * 120, "\n")  # Divider

    # Print missing values
    print("Missing values in each column:\n")
    print(df.isnull().sum(), "\n")
    print("-" * 120, "\n")  # Divider

    return df



df = load_data("usa_rain_prediction_dataset_2024_2025.csv")
with open("imported_data.pkl", "wb") as f:
    pickle.dump(df, f)
print("Imported data saved to imported_data.pkl")

#!/usr/bin/env python3
import sys
import os
import pickle
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.utils import resample


# Preprocessing
def preprocess_data(df):
    df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity']
    df['Wind_Cloud_Ratio'] = df['Wind Speed'] / df['Cloud Cover']

    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.drop('Rain Tomorrow')
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    print("\nData preprocessing is complete.\n")
    df.to_csv('preprocessed_dataset.csv', index=False)
    print("... Saved preprocessed dataset as 'preprocessed_dataset.csv'\n")

    # Print first 5 rows after preprocessing
    print("First 5 rows of the preprocessed dataset:\n")
    print(df.head(), "\n")

    print("-" * 120, "\n")  # Divider

    return df

# Handle class imbalance
def balance_classes(data):
    # Print the original class distribution
    print("Original Class Distribution:\n")
    print(data['Rain Tomorrow'].value_counts(), "\n")
    print("-" * 120, "\n")  # Divider

    # Separate the majority and minority classes
    majority_class = data[data['Rain Tomorrow'] == 0]
    minority_class = data[data['Rain Tomorrow'] == 1]
    print('minority_class:',minority_class, "\n")


    # Upsample the minority class
    minority_upsampled = resample(minority_class,replace=True,n_samples=len(majority_class),random_state=42)
    print('minority_upsampled:',minority_upsampled, "\n")
    # Combine majority class with the upsampled minority class
    data_balanced = pd.concat([majority_class, minority_upsampled])

    print('data_balanced:',data_balanced, "\n")

    # Print the new class distribution
    print("New Class Distribution after Oversampling:\n")
    print(data_balanced['Rain Tomorrow'].value_counts(), "\n")

    return data_balanced


with open("imported_data.pkl", "rb") as f:
    df = pickle.load(f)
    df = preprocess_data(df)
    df_balanced = balance_classes(df)

    print('df_balanced:',df_balanced, "\n")

    
with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump(df_balanced, f)
print("Preprocessed data saved to preprocessed_data.pkl")

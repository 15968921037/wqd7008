#!/usr/bin/env python3
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
import os
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve



def print_section_title(title):
    # Define the width of the box
    box_width = 120

    # Print the top border
    print("─" * box_width)

    # Print the title with padding for centering
    title_line = f" {title} "
    print(f"{title_line.center(box_width, ' ')}")

    # Print the bottom border
    print("─" * box_width)

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

# Plot and save countplot
def plot_countplot(df):
    plt.figure()
    sns.countplot(x='Rain Tomorrow', data=df, palette='coolwarm')
    plt.title('Distribution of Rain Tomorrow')
    plt.savefig('EDA_distribution_rain_tomorrow.png')
    print("... Saved countplot as 'EDA_distribution_rain_tomorrow.png'\n")

    print("-" * 120, "\n")  # Divider

# Plot and save pairplot
def plot_pairplot(df):
    plt.figure()
    pairplot = sns.pairplot(
        df[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Rain Tomorrow']],
        hue='Rain Tomorrow',
        palette='coolwarm'
    )
    pairplot.fig.suptitle("Feature Relationships with Rain Tomorrow Outcome", y=1.02)  # Add title
    plt.savefig('EDA_pairplot_features.png')
    print("... Saved pairplot as 'EDA_pairplot_features.png'\n")

    print("-" * 120, "\n")  # Divider

# Plot and save heatmap
def plot_heatmap(numeric_df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Features')
    plt.savefig('EDA_correlation_heatmap.png')
    print("... Saved heatmap as 'EDA_correlation_heatmap.png'\n")

    print("Data exploration is complete.\n")


with open("preprocessed_data.pkl", "rb") as f:
    df = pickle.load(f)
    plot_countplot(df)
    plot_pairplot(df)
    numeric_df = df[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure']]
    plot_heatmap(numeric_df)

    print_section_title("DATA PREPROCESSING")

    
with open("eda_data.pkl", "wb") as f:
    pickle.dump(numeric_df, f)
print("saved to eda_data.pkl")

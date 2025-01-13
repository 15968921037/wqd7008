#!/usr/bin/env python3

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
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

    # Upsample the minority class
    minority_upsampled = resample(minority_class,
                                  replace=True,
                                  n_samples=len(majority_class),
                                  random_state=42)

    # Combine majority class with the upsampled minority class
    data_balanced = pd.concat([majority_class, minority_upsampled])

    # Print the new class distribution
    print("New Class Distribution after Oversampling:\n")
    print(data_balanced['Rain Tomorrow'].value_counts(), "\n")

    return data_balanced

# Train and evaluate models
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"\n\033[1m{name}\033[0m Cross-Validation Results:\n")
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}\n")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        print(f"{name} Test Set Results:\n")
        print(classification_report(y_test, y_pred))
        print(f"F1-Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC: {roc_auc:.4f}")

        results[name] = accuracy

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"{name.replace(' ', '_')}_confusion_matrix.png")

        print("\n")  # Enter a line

        print(f"... Saved {name} confusion matrix as '{name.replace(' ', '_')}_confusion_matrix.png'\n")

        # Save ROC curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{name} ROC Curve")
            plt.legend()
            plt.savefig(f"{name.replace(' ', '_')}_roc_curve.png")
            print(f"... Saved {name} ROC curve as '{name.replace(' ', '_')}_roc_curve.png'\n")

        print("-" * 120, "\n")  # Divider

    return results

# Main execution
def main():
    print_section_title("EXPLORATORY DATA ANALYSIS")

    df = load_data("usa_rain_prediction_dataset_2024_2025.csv")
    plot_countplot(df)
    plot_pairplot(df)
    numeric_df = df[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure']]
    plot_heatmap(numeric_df)

    print_section_title("DATA PREPROCESSING")

    df = preprocess_data(df)
    df_balanced = balance_classes(df)

    print_section_title("MODEL TRAINING AND EVALUATION")

    le = LabelEncoder()
    df_balanced['Rain Tomorrow'] = le.fit_transform(df_balanced['Rain Tomorrow'])

    X = df_balanced.drop(columns=['Rain Tomorrow', 'Date', 'Location'])
    y = df_balanced['Rain Tomorrow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = train_models(X_train, X_test, y_train, y_test)

    print("Model Performance Summary:\n")
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.4f}")

    print("\nModel training and evaluation is complete.\n")
    print("-" * 120)  # Divider

if __name__ == "__main__":
    main()
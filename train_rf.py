#!/usr/bin/env python3
import pandas as pd
import numpy as np
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

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"\n\033[1m{name}\033[0m Training and Evaluation:\n")


        model.fit(X_train, y_train)
        

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None


        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} F1-Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"{name} ROC-AUC: {roc_auc:.4f}")

        # save result
        results[name] = {
            "Accuracy": accuracy,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        }

        # save matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"{name.replace(' ', '_')}_confusion_matrix.png")
        print(f"... Saved {name} confusion matrix as '{name.replace(' ', '_')}_confusion_matrix.png'\n")

        # save ROC
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



with open("split_data.pkl", "rb") as f:
    split_data = pickle.load(f)
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    
    #Make sure the data format is a NumPy array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Train the model and get the results
    results = train_models(X_train, X_test, y_train, y_test)
    
    #print result
    print("Model Performance Summary:\n")

with open("train_rf.pkl", "wb") as f:
    pickle.dump(results, f)
print("saved to train_rf.pkl")

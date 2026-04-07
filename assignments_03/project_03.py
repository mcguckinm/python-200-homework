import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

os.makedirs("outputs", exist_ok=True)


def print_headers(title):
    print("\n"+ "="*80)
    print(title)
    print("="*80)

def print_top_feature_importances(model, feature_names, top_n=10, label="Model"):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\nTop {top_n} features for {label}: ")
    print(importances.head(top_n))
    return importances

print_headers("Task 1")

spam = fetch_openml(name="spambase", version=1, as_frame=True)

df = spam.frame.copy()

target_col = spam.target.name if hasattr(spam.target, "name") and spam.target.name else df.columns[-1]

df[target_col] = df[target_col].astype(int)

feature_names = [col for col in df.columns if col != target_col]

x = df.drop(columns=[target_col])
y=df[target_col]

print("Dataset shape", df.shape)
print("Number of emails: ", len(df))
print("\nClass counts: ")
print(y.value_counts().sort_index())
print("\nClass Proportions: ")
print(y.value_counts(normalize=True).sort_index())

majority_class_rate = y.value_counts(normalize=True).max()
print(f"\nBaseline accuracy from always predicting the majority class: {majority_class_rate:.4f}")

print(f"\nFeature names preview: {feature_names[:10]}" )


free_col = "word_freq_free"
bang_col = "char_freq_!"
total_caps_col = "capital_run_length_total"

for required_cols in [free_col, bang_col, total_caps_col]:
    if required_cols not in df.columns:
        raise ValueError(
            f"Expected column '{required_cols}' was not found"
            f"Available columns include: {list(df.columns[:15])}"
        )
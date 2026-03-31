"""
train.py — Train the AI vs Human Text Detector
================================================
Dataset: "AI vs Human Text" by Shane Gerami (Kaggle)
URL: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

Steps:
  1. Load dataset (CSV with 'text' and 'generated' columns)
  2. Extract handcrafted linguistic features
  3. Train a Random Forest classifier
  4. Evaluate and save the model
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from features import extract_features


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_PATH   = "data/AI_Human.csv"      # adjust if your CSV has a different name
MODEL_PATH  = "model/detector.pkl"
SAMPLE_SIZE = 20_000                   # use a subset so training is fast on any machine
RANDOM_SEED = 42


def load_data(path: str, sample_size: int) -> pd.DataFrame:
    print(f"[1/4] Loading data from: {path}")
    df = pd.read_csv(path)

    # ── normalise column names ──────────────────
    df.columns = df.columns.str.strip().str.lower()

    # expected columns: 'text', 'generated'  (0 = human, 1 = AI)
    required = {"text", "generated"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must have columns {required}. Found: {list(df.columns)}"
        )

    df = df[["text", "generated"]].dropna()
    df["generated"] = df["generated"].astype(int)

    # balanced sample
    n_each = sample_size // 2
    human = df[df["generated"] == 0].sample(n=min(n_each, len(df[df["generated"]==0])), random_state=RANDOM_SEED)
    ai    = df[df["generated"] == 1].sample(n=min(n_each, len(df[df["generated"]==1])), random_state=RANDOM_SEED)
    df    = pd.concat([human, ai]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"    Loaded {len(df)} samples  (human={len(human)}, AI={len(ai)})")
    return df


def build_features(df: pd.DataFrame):
    print("[2/4] Extracting features …")
    X = np.array([extract_features(t) for t in df["text"]])
    y = df["generated"].values
    print(f"    Feature matrix shape: {X.shape}")
    return X, y


def train_model(X_train, y_train):
    print("[3/4] Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test):
    print("[4/4] Evaluating …")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n    Accuracy : {acc:.4f}")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
    print("    Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"      Human predicted Human : {cm[0][0]:>6}")
    print(f"      Human predicted AI    : {cm[0][1]:>6}")
    print(f"      AI    predicted Human : {cm[1][0]:>6}")
    print(f"      AI    predicted AI    : {cm[1][1]:>6}")


def save_model(clf, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n    Model saved → {path}")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Dataset not found at '{DATA_PATH}'")
        print("  1. Download from https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text")
        print("  2. Place the CSV inside a 'data/' folder in this directory.")
        sys.exit(1)

    df              = load_data(DATA_PATH, SAMPLE_SIZE)
    X, y            = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    clf = train_model(X_train, y_train)
    evaluate(clf, X_test, y_test)
    save_model(clf, MODEL_PATH)
    print("\nDone! Run  python predict.py  to test on new text.\n")


if __name__ == "__main__":
    main()

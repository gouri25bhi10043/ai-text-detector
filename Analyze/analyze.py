"""
analyze.py — Exploratory analysis & feature importance visualisation
=====================================================================
Run AFTER training:
    python analyze.py

Outputs:
  • Prints descriptive statistics comparing AI vs Human text features
  • Saves a bar chart of feature importances → results/feature_importance.png
  • Saves a comparison table                 → results/feature_comparison.csv
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from features import extract_features, FEATURE_NAMES

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False
    print("  (matplotlib not installed — skipping chart; run: pip install matplotlib)")


MODEL_PATH   = "model/detector.pkl"
DATA_PATH    = "data/AI_Human.csv"
RESULTS_DIR  = "results"
SAMPLE_SIZE  = 2000   # small sample for fast analysis
RANDOM_SEED  = 42


def load_sample(path, n):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df[["text", "generated"]].dropna()
    df["generated"] = df["generated"].astype(int)

    n_each = n // 2
    human = df[df["generated"] == 0].sample(n=min(n_each, sum(df["generated"]==0)), random_state=RANDOM_SEED)
    ai    = df[df["generated"] == 1].sample(n=min(n_each, sum(df["generated"]==1)), random_state=RANDOM_SEED)
    return pd.concat([human, ai]).reset_index(drop=True)


def build_feature_df(df):
    feat_list = [extract_features(t) for t in df["text"]]
    feat_df   = pd.DataFrame(feat_list, columns=FEATURE_NAMES)
    feat_df["label"] = df["generated"].map({0: "Human", 1: "AI"})
    return feat_df


def print_comparison(feat_df):
    print("\n" + "═" * 70)
    print("  Feature Comparison: Human vs AI  (mean values)")
    print("═" * 70)
    print(f"  {'Feature':<26}  {'Human':>10}  {'AI':>10}  {'Diff %':>8}")
    print("  " + "-" * 60)
    for feat in FEATURE_NAMES:
        h_mean = feat_df[feat_df["label"] == "Human"][feat].mean()
        a_mean = feat_df[feat_df["label"] == "AI"  ][feat].mean()
        diff   = ((a_mean - h_mean) / max(abs(h_mean), 1e-9)) * 100
        marker = "◄" if abs(diff) > 20 else ""
        print(f"  {feat:<26}  {h_mean:>10.4f}  {a_mean:>10.4f}  {diff:>+7.1f}%  {marker}")
    print("═" * 70)


def plot_feature_importance(clf):
    if not MATPLOTLIB:
        return

    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in indices]
    sorted_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_names[::-1], sorted_vals[::-1], color="#4C72B0")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importances — AI vs Human Text Detector", fontsize=14)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n  Chart saved → {out_path}")
    plt.close()


def main():
    for path, name in [(DATA_PATH, "Dataset"), (MODEL_PATH, "Model")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at '{path}'")
            if name == "Model":
                print("  Run  python train.py  first.")
            sys.exit(1)

    print("[1/3] Loading sample …")
    df = load_sample(DATA_PATH, SAMPLE_SIZE)

    print("[2/3] Extracting features …")
    feat_df = build_feature_df(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "feature_comparison.csv")
    (feat_df.groupby("label")[FEATURE_NAMES]
            .mean()
            .T
            .to_csv(csv_path))
    print(f"  Comparison table saved → {csv_path}")

    print_comparison(feat_df)

    print("[3/3] Plotting feature importances …")
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    plot_feature_importance(clf)
    print("\nAnalysis complete.\n")


if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""
Retrain symptom-based malaria classifiers and export refreshed joblib artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET_COL = "severe_maleria"
MODEL_SPECS = {
    "malaria_symptom_decision_tree.joblib": DecisionTreeClassifier(
        criterion="gini", max_depth=6, random_state=42
    ),
    "malaria_symptom_svm.joblib": SVC(
        kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42
    ),
    "malaria_symptom_logistic_regression.joblib": LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=42
    ),
    "malaria_symptom_random_forest.joblib": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42
    ),
}


def load_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from dataset.")

    feature_names = [col for col in df.columns if col != TARGET_COL]
    X = df[feature_names].astype(np.float32)
    y = df[TARGET_COL].astype(int)
    return X, y, feature_names


def save_artifact(obj, path: Path) -> None:
    joblib.dump(obj, path)
    print(f"[saved] {path.name}")


def train_models(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    for filename, model in MODEL_SPECS.items():
        if "svm" in filename or "logistic" in filename:
            X_tr = scaler.transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n[{filename}] accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred, digits=3))

        save_artifact(model, output_dir / filename)

    save_artifact(scaler, output_dir / "malaria_symptom_scaler.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain malaria symptom classifiers.")
    parser.add_argument(
        "--data",
        default="mmc1.csv",
        type=Path,
        help="Path to symptom dataset CSV (default: mmc1.csv)",
    )
    parser.add_argument(
        "--output",
        default=".",
        type=Path,
        help="Directory where joblib artifacts will be stored (default: project root)",
    )
    args = parser.parse_args()

    output_dir = args.output if isinstance(args.output, Path) else Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_dataset(args.data)
    save_artifact(feature_names, output_dir / "malaria_symptom_features.joblib")

    train_models(X, y, output_dir)
    print("\nTraining complete. Artifacts refreshed.")


if __name__ == "__main__":
    main()


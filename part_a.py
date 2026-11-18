"""
Part A: dish category classifier.

TF-IDF (word + character n-grams) into Logistic Regression, trained on
dishes.csv, evaluated on a held-out stratified split, then used to predict
categories for dish names it has never seen.

Run:  python part_a.py
"""

from __future__ import annotations

import json
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load and clean. The files we were given are tidy, but the cleaning is
#    written defensively because real menu exports are not: spacing, casing,
#    duplicate rows and stray categories all appear in practice.
# ---------------------------------------------------------------------------
def load_dishes(path: str = "data/dishes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    df["dish_name"] = (
        df["dish_name"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["category"] = (
        df["category"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.title()
    )

    df = df.dropna(subset=["dish_name", "category"])
    df = df[df["dish_name"].str.len() > 0]
    df = df.drop_duplicates(subset=["dish_name"])        # one label per dish

    # collapse obvious near-duplicate labels (e.g. "main course" / "Main Course")
    keep = {"Starter", "Main Course", "Dessert", "Bread", "Rice", "Beverage"}
    df["category"] = df["category"].where(df["category"].isin(keep), df["category"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
def build_model() -> Pipeline:
    # Word n-grams catch "chicken biryani"; character n-grams survive typos and
    # Indian-English spelling variants ("panner", "masla"), which is what the
    # future inputs will actually look like.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
    )
    from sklearn.pipeline import FeatureUnion
    features = FeatureUnion([("word", vectorizer), ("char", char_vectorizer)])

    clf = LogisticRegression(
        max_iter=2000,
        C=10.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("features", features), ("clf", clf)])


# ---------------------------------------------------------------------------
# 3. Train, evaluate, report
# ---------------------------------------------------------------------------
def main() -> dict:
    df = load_dishes()
    X, y = df["dish_name"].values, df["category"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cv = cross_val_score(build_model(), X, y, cv=5, scoring="accuracy")

    print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")
    print(f"Test accuracy: {acc:.3f}")
    print(f"5-fold CV accuracy: {cv.mean():.3f} (+/- {cv.std():.3f})")
    print("\nPer-category report:\n")
    print(report)
    print("Confusion matrix (rows = true, cols = predicted)")
    print("labels:", labels)
    print(cm)

    # -----------------------------------------------------------------------
    # 4. Five unseen dishes
    # -----------------------------------------------------------------------
    unseen = [
        "Paneer Kathi Roll",
        "Mango Kulfi Falooda",
        "Tandoori Broccoli",
        "Veg Hakka Noodles",
        "Rose Lassi",
    ]
    proba = model.predict_proba(unseen)
    classes = model.classes_
    predictions = []
    print("\nUnseen dishes:")
    for name, row in zip(unseen, proba):
        order = row.argsort()[::-1]
        top, conf = classes[order[0]], row[order[0]]
        second, conf2 = classes[order[1]], row[order[1]]
        predictions.append({
            "dish": name, "predicted": top, "confidence": round(float(conf), 3),
            "runner_up": second, "runner_up_confidence": round(float(conf2), 3),
        })
        print(f"  {name:22s} -> {top:12s} ({conf:.2f})   second: {second} ({conf2:.2f})")

    out = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_accuracy": round(float(acc), 4),
        "cv_accuracy_mean": round(float(cv.mean()), 4),
        "cv_accuracy_std": round(float(cv.std()), 4),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_category_report": report,
        "unseen_predictions": predictions,
        "n_dishes": int(len(df)),
    }
    with open("part_a_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # expose the fitted model for the notebook / demo
    import joblib
    joblib.dump(model, "category_model.joblib")
    return out


def predict_category(dish: str, model_path: str = "category_model.joblib") -> str:
    """Single-dish helper used in the write-up and the demo."""
    import joblib
    model = joblib.load(model_path)
    return str(model.predict([dish])[0])


if __name__ == "__main__":
    main()

"""Melatih model ringkas prediksi hujan besok (RainTomorrow).

Mengikuti alur notebook: buang baris tanpa target, drop duplikat, split stratified
80/20 (random_state=42), gradient boosting untuk data tabular. Bedanya, model ini
hanya memakai 6 fitur paling berpengaruh + Location agar mudah diisi di aplikasi.
Model: XGBoost (sama seperti model terbaik di notebook).

Jalankan:  python train_model.py
Output  :  model.joblib
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

CSV_PATH = "weatherAUS.csv"
MODEL_PATH = "model.joblib"

# 6 fitur terpilih (berdasarkan permutation importance ROC-AUC; Pressure9am dibuang
# karena redundan dengan Pressure3pm). Rainfall = curah hujan hari ini.
FEATURES = ["Humidity3pm", "Pressure3pm", "WindGustSpeed",
            "Sunshine", "Rainfall", "Cloud3pm"]


def build_pipeline():
    # Location di-one-hot (seperti notebook); fitur numerik dibiarkan apa adanya,
    # XGBoost menangani nilai kosong sendiri.
    pre = ColumnTransformer(
        [("loc", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Location"])],
        remainder="passthrough")
    model = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.8, reg_lambda=5,
        tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=42)
    return Pipeline([("prep", pre), ("model", model)])


def train(csv_path=CSV_PATH, model_path=MODEL_PATH, verbose=True):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["RainTomorrow"]).drop_duplicates().copy()
    df["y"] = (df["RainTomorrow"] == "Yes").astype(int)

    X, y = df[["Location"] + FEATURES], df["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    # validasi terpisah untuk memilih threshold keputusan
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)

    locations = sorted(df["Location"].unique())
    pipe = build_pipeline().fit(X_fit, y_fit)

    # threshold dengan F1 terbaik di validation (kelas hujan hanya ~22%)
    p_val = pipe.predict_proba(X_val)[:, 1]
    grid = np.arange(0.15, 0.70, 0.01)
    threshold = float(grid[np.argmax([f1_score(y_val, p_val >= t) for t in grid])])

    p_te = pipe.predict_proba(X_te)[:, 1]
    pred = p_te >= threshold
    metrics = {
        "roc_auc": roc_auc_score(y_te, p_te),
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "threshold": threshold,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }

    imp = permutation_importance(pipe, X_te, y_te, scoring="roc_auc",
                                 n_repeats=3, random_state=0)
    importance = {k: float(v) for k, v in zip(["Location"] + FEATURES, imp.importances_mean.round(4))}

    # model final dilatih ulang pada seluruh data latih (train+val)
    pipe = build_pipeline().fit(X_tr, y_tr)

    global_med = df[FEATURES].median()
    loc_med = df.groupby("Location")[FEATURES].median()
    loc_med = loc_med.fillna(global_med)
    loc_rate = df.groupby("Location")["y"].mean().round(3).to_dict()

    joblib.dump({
        "pipeline": pipe, "features": FEATURES, "locations": locations,
        "threshold": threshold, "metrics": metrics, "importance": importance,
        "location_defaults": loc_med.round(1).to_dict(orient="index"),
        "location_rain_rate": loc_rate,
    }, model_path)

    if verbose:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})
        print("Importance:", importance)
    return model_path


if __name__ == "__main__":
    train()

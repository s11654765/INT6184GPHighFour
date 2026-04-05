# -*- coding: utf-8 -*-
"""
在 try/final_model/model_final.ipynb 同一数据与编码上，使用 GridSearch 得到的最优 XGBoost 超参
（n_estimators=200, max_depth=6, learning_rate=0.1）在全量数据上训练并保存 joblib。

数据文件：try/final_model/Obesity_data_clean.csv（与 notebook 同目录相对路径）。
随机种子与 notebook 一致：RANDOM_STATE = 2。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import joblib

try:
    from xgboost import XGBClassifier
except ImportError:
    print("Install: pip install xgboost pandas scikit-learn", file=sys.stderr)
    raise

from model_final_encode import DROP_COLS, SEVERITY_ORDER, transform_obesity_features

# 与 model_final.ipynb 中 GridSearchCV 输出一致（Best params）
RANDOM_STATE = 2
XGB_BEST = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "eval_metric": "mlogloss",
    "n_jobs": 1,
}


def main() -> None:
    base = Path(__file__).resolve().parent
    root = base.parent
    data_path = root / "try" / "final_model" / "Obesity_data_clean.csv"
    if not data_path.is_file():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(data_path)
    if "obesity_level" not in df.columns:
        sys.exit("Column obesity_level missing.")

    df["obesity_level"] = pd.Categorical(
        df["obesity_level"], categories=SEVERITY_ORDER, ordered=True
    )

    X_raw = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df["obesity_level"]

    X = transform_obesity_features(X_raw)
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(np.int8)

    le = LabelEncoder()
    le.fit(SEVERITY_ORDER)
    y_int = le.transform(y.astype(str))

    try:
        model = XGBClassifier(**XGB_BEST, use_label_encoder=False)
    except TypeError:
        model = XGBClassifier(**XGB_BEST)

    model.fit(X, y_int)

    out_path = base / "xgb_model_bundle.joblib"
    bundle = {
        "model": model,
        "label_encoder": le,
        "feature_columns": list(X.columns),
        "severity_order": SEVERITY_ORDER,
        "xgb_params": {**XGB_BEST, "source": "model_final.ipynb GridSearch best"},
        "data_source": str(data_path),
    }
    joblib.dump(bundle, out_path)
    print(f"Saved {out_path}, n_samples={len(X)}, n_features={X.shape[1]}")


if __name__ == "__main__":
    main()

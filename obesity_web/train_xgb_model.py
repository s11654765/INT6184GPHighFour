# -*- coding: utf-8 -*-
"""
使用与 model_final.ipynb 相同的 XGBoost 超参数，在 Obesity_Data_clean_onehot.xlsx 全量数据上训练，
并保存 joblib 供 Flask 加载。部署推理通常使用全量数据重训；若需与某次 holdout 完全一致，请在笔记本中导出模型。
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
    print("请先安装: pip install xgboost pandas openpyxl scikit-learn", file=sys.stderr)
    raise

from feature_mappings import FEATURE_COLUMNS, SEVERITY_ORDER

RANDOM_STATE = 42

# 与 model_final.ipynb「XGBoost（最优参数固定）」一致
XGB_SEARCH_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}
XGB_FIXED = {
    "random_state": RANDOM_STATE,
    "eval_metric": "mlogloss",
    "n_jobs": -1,
}


def main() -> None:
    base = Path(__file__).resolve().parent
    root = base.parent
    data_path = root / "Obesity_Data_clean_onehot.xlsx"
    if not data_path.is_file():
        print(f"未找到数据文件: {data_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(data_path)
    if "obesity_level" not in df.columns:
        sys.exit("数据缺少列 obesity_level")

    X = df.drop(columns=["obesity_level"])
    y = df["obesity_level"]

    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(np.int8)

    # 列顺序固定，与线上一致
    X = X[FEATURE_COLUMNS]

    le = LabelEncoder()
    le.fit(SEVERITY_ORDER)
    y_int = le.transform(y.astype(str))

    try:
        model = XGBClassifier(**XGB_FIXED, **XGB_SEARCH_PARAMS, use_label_encoder=False)
    except TypeError:
        model = XGBClassifier(**XGB_FIXED, **XGB_SEARCH_PARAMS)

    model.fit(X, y_int)

    out_path = base / "xgb_model_bundle.joblib"
    bundle = {
        "model": model,
        "label_encoder": le,
        "feature_columns": FEATURE_COLUMNS,
        "severity_order": SEVERITY_ORDER,
        "xgb_params": {**XGB_FIXED, **XGB_SEARCH_PARAMS},
    }
    joblib.dump(bundle, out_path)
    print(f"已保存: {out_path}，样本数 {len(X)}，特征数 {X.shape[1]}")


if __name__ == "__main__":
    main()

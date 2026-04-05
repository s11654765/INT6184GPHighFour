# -*- coding: utf-8 -*-
"""
与 try/final_model/model_final.ipynb 中 transform_obesity_features 一致的编码逻辑。
原始特征列（DROP_COLS 之后）为：
age, high_cal_food, veg_consumption, meals_per_day, snacking, smoking, water_intake,
calorie_monitor, physical_activity, screen_time, alcohol, transport
"""
from __future__ import annotations

import pandas as pd

SEVERITY_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

# 与 notebook 一致
DROP_COLS = ["gender", "height", "weight", "family_history", "obesity_level"]


def transform_obesity_features(X: pd.DataFrame) -> pd.DataFrame:
    """与 model_final.ipynb 中函数一致。"""
    out = pd.DataFrame(index=X.index)
    if "age" in X.columns:
        out["age"] = pd.to_numeric(X["age"], errors="coerce")

    def bin_yes_no(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().map({"yes": 1, "no": 0})

    out["high_cal_food"] = bin_yes_no(X["high_cal_food"])
    out["smoking"] = bin_yes_no(X["smoking"])
    out["calorie_monitor"] = bin_yes_no(X["calorie_monitor"])

    freq_order = ["no", "Sometimes", "Frequently", "Always"]
    fo = {k: i for i, k in enumerate(freq_order)}
    out["snacking"] = X["snacking"].map(fo)
    out["alcohol"] = X["alcohol"].map(fo)

    out["veg_consumption"] = X["veg_consumption"].round().clip(1, 3).astype(int)
    out["meals_per_day"] = X["meals_per_day"].round().clip(1, 4).astype(int)
    out["water_intake"] = X["water_intake"].round().clip(1, 3).astype(int)
    out["physical_activity"] = X["physical_activity"].round().clip(0, 3).astype(int)
    out["screen_time"] = X["screen_time"].round().clip(0, 2).astype(int)

    transport_dum = pd.get_dummies(X["transport"], prefix="transport", dtype=int)
    out = pd.concat([out, transport_dum], axis=1)
    return out


def raw_row_to_encoded_frame(
    *,
    age: float,
    high_cal_food: str | int,
    veg_consumption: float,
    meals_per_day: float,
    snacking: str,
    smoking: str,
    water_intake: float,
    calorie_monitor: str | int,
    physical_activity: float,
    screen_time: float,
    alcohol: str,
    transport: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """单行原始输入 → 编码后与训练列对齐的 DataFrame。"""
    def yn(v) -> str:
        if isinstance(v, (int, float)):
            return "yes" if int(v) == 1 else "no"
        return str(v).strip().lower()

    row = {
        "age": age,
        "high_cal_food": yn(high_cal_food),
        "veg_consumption": float(veg_consumption),
        "meals_per_day": float(meals_per_day),
        "snacking": str(snacking).strip(),
        "smoking": yn(smoking),
        "water_intake": float(water_intake),
        "calorie_monitor": yn(calorie_monitor),
        "physical_activity": float(physical_activity),
        "screen_time": float(screen_time),
        "alcohol": str(alcohol).strip(),
        "transport": str(transport).strip(),
    }
    X_raw = pd.DataFrame([row])
    X_enc = transform_obesity_features(X_raw)
    return X_enc.reindex(columns=feature_columns, fill_value=0)

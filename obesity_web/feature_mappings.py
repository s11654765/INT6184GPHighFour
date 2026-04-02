# -*- coding: utf-8 -*-
"""
与 Obesity_Data_clean_onehot.xlsx / obesity database.py 一致的编码规则。
transport：参照编码 0=公共交通(基准)、1=步行、2=汽车、3=自行车 → 仅 transport_1/2/3 三列（drop-first）。
"""
from __future__ import annotations

import numpy as np
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

# 课堂展示用中文名（与英文类名一一对应）
OBESITY_LEVEL_ZH = {
    "Insufficient_Weight": "体重不足",
    "Normal_Weight": "正常体重",
    "Overweight_Level_I": "超重 I 级",
    "Overweight_Level_II": "超重 II 级",
    "Obesity_Type_I": "肥胖 I 型",
    "Obesity_Type_II": "肥胖 II 型",
    "Obesity_Type_III": "肥胖 III 型",
}

# obesity database.py：零食 / 饮酒
SNACKING_ALCOHOL_MAP = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3,
}

# 是/否类
YES_NO_MAP = {"no": 0, "yes": 1}

# transport → one-hot（基准类 公共交通 全 0）
TRANSPORT_KEYS = (
    "Public_Transportation",
    "Walking",
    "Automobile",
    "Bike",
)


def transport_to_dummies(transport: str) -> tuple[int, int, int]:
    if transport not in TRANSPORT_KEYS:
        raise ValueError(f"transport 必须是 {TRANSPORT_KEYS} 之一，收到: {transport!r}")
    # 与 Excel 中验证一致：Walking→(1,0,0), Automobile→(0,1,0), Bike→(0,0,1), 公交→(0,0,0)
    return {
        "Public_Transportation": (0, 0, 0),
        "Walking": (1, 0, 0),
        "Automobile": (0, 1, 0),
        "Bike": (0, 0, 1),
    }[transport]


FEATURE_COLUMNS = [
    "gender",
    "age",
    "height",
    "weight",
    "family_history",
    "high_cal_food",
    "snacking",
    "smoking",
    "physical_activity",
    "screen_time",
    "alcohol",
    "transport_1",
    "transport_2",
    "transport_3",
]


def build_feature_frame(
    *,
    gender: int,
    age: float,
    height: float,
    weight: float,
    family_history: int,
    high_cal_food: int,
    snacking: int,
    smoking: int,
    physical_activity: int,
    screen_time: int,
    alcohol: int,
    transport: str,
) -> pd.DataFrame:
    """单行特征，列顺序与训练数据一致。"""
    t1, t2, t3 = transport_to_dummies(transport)
    row = {
        "gender": int(gender),
        "age": int(round(age)),
        "height": round(float(height), 2),
        "weight": round(float(weight), 2),
        "family_history": int(family_history),
        "high_cal_food": int(high_cal_food),
        "snacking": int(snacking),
        "smoking": int(smoking),
        "physical_activity": int(round(physical_activity)),
        "screen_time": int(round(screen_time)),
        "alcohol": int(alcohol),
        "transport_1": np.int8(t1),
        "transport_2": np.int8(t2),
        "transport_3": np.int8(t3),
    }
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return X

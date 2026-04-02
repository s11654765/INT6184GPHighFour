# -*- coding: utf-8 -*-
"""
肥胖等级预测：Flask 提供静态页 + /api/predict。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import joblib

from feature_mappings import (
    OBESITY_LEVEL_ZH,
    SNACKING_ALCOHOL_MAP,
    build_feature_frame,
)


def _base_dir() -> Path:
    """开发时：脚本目录；PyInstaller 打包后：解压目录 sys._MEIPASS。"""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


BASE = _base_dir()
BUNDLE_PATH = BASE / "xgb_model_bundle.joblib"

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="")


def load_bundle():
    if not BUNDLE_PATH.is_file():
        print(
            f"错误: 未找到模型文件 {BUNDLE_PATH}，请先运行: py train_xgb_model.py",
            file=sys.stderr,
        )
        return None
    return joblib.load(BUNDLE_PATH)


BUNDLE = load_bundle()


def _parse_yes_no(v) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return int(v)
    s = str(v).strip().lower()
    if s in ("0", "no", "否", "n"):
        return 0
    if s in ("1", "yes", "是", "y"):
        return 1
    raise ValueError("yes/no 字段应为 0/1 或 yes/no/是/否")


def _parse_snack_alcohol(v) -> int:
    if isinstance(v, (int, float)) and int(v) in (0, 1, 2, 3):
        return int(v)
    s = str(v).strip()
    if s in SNACKING_ALCOHOL_MAP:
        return SNACKING_ALCOHOL_MAP[s]
    raise ValueError("零食/饮酒频率应为 0–3 或 no/Sometimes/Frequently/Always")


@app.route("/")
def index():
    return send_from_directory(BASE / "static", "index.html")


@app.route("/api/health")
def health():
    ok = BUNDLE is not None
    return jsonify({"ok": ok, "model_loaded": ok})


@app.route("/api/predict", methods=["POST"])
def predict():
    if BUNDLE is None:
        return jsonify({"error": "模型未加载，请先运行 train_xgb_model.py"}), 503

    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "无效的 JSON"}), 400

    try:
        gender = int(data["gender"])
        if gender not in (0, 1):
            raise ValueError("gender 应为 0(女) 或 1(男)")

        age = float(data["age"])
        height = float(data["height"])
        weight = float(data["weight"])
        family_history = _parse_yes_no(data["family_history"])
        high_cal_food = _parse_yes_no(data["high_cal_food"])
        smoking = _parse_yes_no(data["smoking"])
        snacking = _parse_snack_alcohol(data["snacking"])
        alcohol = _parse_snack_alcohol(data["alcohol"])
        physical_activity = int(round(float(data["physical_activity"])))
        screen_time = int(round(float(data["screen_time"])))
        transport = str(data["transport"]).strip()

        X = build_feature_frame(
            gender=gender,
            age=age,
            height=height,
            weight=weight,
            family_history=family_history,
            high_cal_food=high_cal_food,
            snacking=snacking,
            smoking=smoking,
            physical_activity=physical_activity,
            screen_time=screen_time,
            alcohol=alcohol,
            transport=transport,
        )
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    model = BUNDLE["model"]
    le = BUNDLE["label_encoder"]

    idx = model.predict(X.values)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X.values)[0].tolist()
    label_en = le.inverse_transform(np.array([int(idx)], dtype=int))[0]
    label_zh = OBESITY_LEVEL_ZH.get(str(label_en), str(label_en))

    classes = [str(c) for c in le.classes_]
    probs = None
    if proba is not None:
        probs = {classes[i]: round(float(proba[i]), 4) for i in range(len(classes))}

    return jsonify(
        {
            "predicted_level_en": str(label_en),
            "predicted_level_zh": label_zh,
            "probabilities": probs,
        }
    )


def _default_port() -> int:
    p = os.environ.get("PORT") or os.environ.get("FLASK_PORT") or "5000"
    try:
        return int(p)
    except ValueError:
        return 5000


if __name__ == "__main__":
    port = _default_port()
    # 打包成 exe 时自动打开浏览器（仅 Windows 常见）
    if getattr(sys, "frozen", False):
        try:
            import webbrowser

            webbrowser.open(f"http://127.0.0.1:{port}/")
        except Exception:
            pass
    app.run(host="0.0.0.0", port=port, debug=False)

# -*- coding: utf-8 -*-
"""
Flask: English UI + /api/predict using model_final.ipynb pipeline (Obesity_data_clean.csv encoding).
Fixed defaults (not on form): age=15, meals_per_day=3.
Model does not use gender / height / weight / family_history (dropped in notebook).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import joblib

from feature_mappings import FREQ_INDEX_TO_STR, SNACKING_ALCOHOL_MAP
from model_final_encode import raw_row_to_encoded_frame
from student_advice_en import advice_for_level

DEFAULT_AGE = 15.0
DEFAULT_MEALS_PER_DAY = 3.0


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


BASE = _base_dir()
BUNDLE_PATH = BASE / "xgb_model_bundle.joblib"

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="")


def load_bundle():
    if not BUNDLE_PATH.is_file():
        print(
            f"Model bundle not found: {BUNDLE_PATH}. Run: py train_xgb_model.py",
            file=sys.stderr,
        )
        return None
    return joblib.load(BUNDLE_PATH)


BUNDLE = load_bundle()


def _parse_yes_no(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)) and v in (0, 1):
        return "yes" if int(v) == 1 else "no"
    s = str(v).strip().lower()
    if s in ("0", "no", "n"):
        return "no"
    if s in ("1", "yes", "y"):
        return "yes"
    raise ValueError("yes/no field must be 0/1 or yes/no")


def _parse_freq(v) -> str:
    if isinstance(v, (int, float)) and int(v) in (0, 1, 2, 3):
        return FREQ_INDEX_TO_STR[int(v)]
    s = str(v).strip()
    if s in SNACKING_ALCOHOL_MAP:
        return s
    raise ValueError("Frequency must be 0–3 or no/Sometimes/Frequently/Always")


def _parse_transport(v) -> str:
    s = str(v).strip()
    allowed = (
        "Public_Transportation",
        "Walking",
        "Automobile",
        "Motorbike",
        "Bike",
    )
    if s not in allowed:
        raise ValueError(f"transport must be one of {allowed}")
    return s


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
        return jsonify({"error": "Model not loaded. Run train_xgb_model.py locally."}), 503

    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        high_cal = _parse_yes_no(data["high_cal_food"])
        veg = float(data["veg_consumption"])
        smoking = _parse_yes_no(data["smoking"])
        snacking = _parse_freq(data["snacking"])
        alcohol = _parse_freq(data["alcohol"])
        water = float(data["water_intake"])
        cal_mon = _parse_yes_no(data["calorie_monitor"])
        physical_activity = float(data["physical_activity"])
        screen_time = float(data["screen_time"])
        transport = _parse_transport(data["transport"])

        X = raw_row_to_encoded_frame(
            age=DEFAULT_AGE,
            high_cal_food=high_cal,
            veg_consumption=veg,
            meals_per_day=DEFAULT_MEALS_PER_DAY,
            snacking=snacking,
            smoking=smoking,
            water_intake=water,
            calorie_monitor=cal_mon,
            physical_activity=physical_activity,
            screen_time=screen_time,
            alcohol=alcohol,
            transport=transport,
            feature_columns=BUNDLE["feature_columns"],
        )
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    model = BUNDLE["model"]
    le = BUNDLE["label_encoder"]

    idx = model.predict(X.values)[0]
    label_key = str(le.inverse_transform(np.array([int(idx)], dtype=int))[0])
    title_en, advice_en = advice_for_level(label_key)

    return jsonify(
        {
            "level_key": label_key,
            "level_title": title_en,
            "advice": advice_en,
            "defaults_used": {
                "age": int(DEFAULT_AGE),
                "meals_per_day": int(DEFAULT_MEALS_PER_DAY),
            },
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
    if getattr(sys, "frozen", False):
        try:
            import webbrowser

            webbrowser.open(f"http://127.0.0.1:{port}/")
        except Exception:
            pass
    app.run(host="0.0.0.0", port=port, debug=False)

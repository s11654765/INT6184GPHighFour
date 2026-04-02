# 一次性脚本：生成 model.ipynb（运行后若不需要可删除本文件）
import json

def cell_md(text: str) -> dict:
    lines = text.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def cell_code(text: str) -> dict:
    lines = text.strip().split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


cells = []

cells.append(
    cell_md(
        """# 肥胖等级多分类（与 `model.py` 同步）

**数据**：`Obesity_Data_clean_onehot.xlsx`（已二值/one-hot 编码，本笔记不再重复编码）。

**用法**：从上到下依次运行各单元格（Kernel → Restart & Run All 可一键跑通）。

**依赖**：`pip install scikit-learn pandas openpyxl xgboost numpy`"""
    )
)

cells.append(
    cell_code(
        """import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None"""
    )
)

cells.append(
    cell_code(
        """# 全局配置（与 model.py 一致）
RANDOM_STATE = 42
TEST_SIZE = 0.2
RUN_REPEATED_HOLDOUT = True
N_REPEATED_HOLDOUT = 3
RUN_TREE_ROUND3 = True
TUNING_PLATEAU_EPS = 0.0008

SEVERITY_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]"""
    )
)

cells.append(
    cell_code(
        '''def _refine_grid_decision_tree(bp):
    """第三轮决策树：在最优参数附近极小网格。"""
    md = bp["max_depth"]
    if md is None:
        depth_opts = [16, 20, None]
    else:
        depth_opts = sorted(
            set([max(4, md - 2), md - 1, md, md + 1, min(md + 2, 30)])
        )
    ccp = bp["ccp_alpha"]
    ccp_opts = sorted({ccp, ccp * 0.5, ccp * 2, 0.0})[:4]
    return {
        "max_depth": depth_opts,
        "min_samples_split": [bp["min_samples_split"]],
        "min_samples_leaf": [bp["min_samples_leaf"]],
        "ccp_alpha": ccp_opts,
        "max_leaf_nodes": [bp["max_leaf_nodes"]],
    }


def _refine_grid_random_forest(bp):
    """第三轮随机森林：微调 n_estimators 与 max_depth。"""
    ne = bp["n_estimators"]
    ne_opts = sorted(set([max(50, ne - 50), ne, ne + 50, ne + 100]))
    md = bp["max_depth"]
    if md is None:
        md_opts = [16, 20, 22, None]
    else:
        # 只对整数深度排序；None 不能与 int 一起 sorted，单独附加表示「不限制深度」
        nums = sorted(set([max(8, md - 3), md, min(md + 3, 35)]))
        md_opts = nums + [None]
    return {
        "n_estimators": ne_opts,
        "max_depth": md_opts,
        "min_samples_split": [bp["min_samples_split"]],
        "min_samples_leaf": [bp["min_samples_leaf"]],
        "max_features": [bp["max_features"]],
    }


def _stability_cv(estimator, X_train, y_train, cv_splitter):
    scores = cross_val_score(
        estimator, X_train, y_train, cv=cv_splitter,
        scoring="f1_macro", n_jobs=-1,
    )
    return {"cv_mean_macro_f1": float(np.mean(scores)), "cv_std_macro_f1": float(np.std(scores))}


def _stability_cv_xgb(xgb_model, X_train, y_train_int, cv_splitter):
    scores = cross_val_score(
        xgb_model, X_train, y_train_int, cv=cv_splitter,
        scoring="f1_macro", n_jobs=-1,
    )
    return {"cv_mean_macro_f1": float(np.mean(scores)), "cv_std_macro_f1": float(np.std(scores))}


def _repeated_holdout_macro_f1(estimator, X, y, n_times, test_size):
    scores = []
    for k in range(n_times):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y,
            random_state=RANDOM_STATE + 17 + k,
        )
        clf = clone(estimator)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        scores.append(f1_score(y_te, pred, average="macro", zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def _repeated_holdout_macro_f1_xgb(X, y, le, n_times, test_size, xgb_params):
    if XGBClassifier is None:
        return 0.0, 0.0
    scores = []
    for k in range(n_times):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y,
            random_state=RANDOM_STATE + 17 + k,
        )
        y_tr_i = le.transform(y_tr.astype(str))
        try:
            model = XGBClassifier(**xgb_params, use_label_encoder=False)
        except TypeError:
            model = XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr_i)
        pred_i = model.predict(X_te)
        pred_cat = le.inverse_transform(pred_i.astype(int))
        pred_cat = pd.Categorical(pred_cat, categories=SEVERITY_ORDER, ordered=True)
        scores.append(f1_score(y_te, pred_cat, average="macro", zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def _collect_result(model_name, best_params, best_cv_score, stability, y_test, y_pred, label_order):
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec_m = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_test, y_pred, average="macro", zero_division=0)
    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=label_order)
    return {
        "model": model_name,
        "best_params": str(best_params),
        "best_cv_macro_f1": round(best_cv_score, 5),
        "cv_mean_macro_f1": stability["cv_mean_macro_f1"],
        "cv_std_macro_f1": stability["cv_std_macro_f1"],
        "test_accuracy": round(acc, 5),
        "test_macro_f1": round(macro_f1, 5),
        "test_weighted_f1": round(weighted_f1, 5),
        "test_precision_macro": round(prec_m, 5),
        "test_recall_macro": round(rec_m, 5),
        "test_precision_weighted": round(prec_w, 5),
        "test_recall_weighted": round(rec_w, 5),
        "confusion_matrix_shape": cm.shape,
    }


def _neighbors_none(val, pool):
    if val is None:
        return [None]
    out = {val}
    for p in pool:
        if p is None:
            continue
        if abs(p - val) <= 6:
            out.add(p)
    out.add(None)
    return list(out)


def _neighbors_int(val, pool):
    out = {int(val)}
    for p in pool:
        if abs(p - val) <= max(3, val // 2):
            out.add(p)
    return sorted(out)'''
    )
)

cells.append(
    cell_md(
        """## 1. 读取数据、划分特征与标签、训练/测试集

（不做 one-hot）打印列名核对；`stratify` 分层划分。"""
    )
)

cells.append(
    cell_code(
        """path = "Obesity_Data_clean_onehot.xlsx"
df = pd.read_excel(path)
print("数据集形状:", df.shape)
print("列名列表:", list(df.columns))

if "obesity_level" not in df.columns:
    raise ValueError("缺少目标列 obesity_level，请检查 Excel 列名。")
df["obesity_level"] = pd.Categorical(
    df["obesity_level"], categories=SEVERITY_ORDER, ordered=True
)

X = df.drop(columns=["obesity_level"])
y = df["obesity_level"]
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(np.int8)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
)
print("训练集:", X_train.shape, "测试集:", X_test.shape)"""
    )
)

cells.append(
    cell_code(
        """# 评价指标：主指标 Macro F1；5 折分层 CV 划分器
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results_rows = []
model_summaries = {}"""
    )
)

cells.append(
    cell_md(
        """## 2. 逻辑回归（Pipeline：StandardScaler + LR，两轮 GridSearch）

在 Pipeline 上搜索，避免先标准化全训练集再 CV 造成泄漏；仅 LR 标准化，树模型用原始 X。"""
    )
)

cells.append(
    cell_code(
        """pipe_lr = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
        ),
    ]
)
param_grid_lr_r1 = [
    {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "newton-cg", "saga"],
    },
    {
        "clf__C": [0.05, 0.1, 0.5, 1.0, 5.0],
        "clf__penalty": ["l1"],
        "clf__solver": ["saga"],
    },
]
grid_lr = GridSearchCV(
    pipe_lr, param_grid_lr_r1, cv=cv_splitter,
    scoring="f1_macro", n_jobs=-1, refit=True,
)
grid_lr.fit(X_train, y_train)

best_c = grid_lr.best_params_.get("clf__C", 1.0)
best_pen = grid_lr.best_params_.get("clf__penalty", "l2")
best_sol = grid_lr.best_params_.get("clf__solver", "lbfgs")
neighbors_c = sorted(
    set([best_c, best_c * 0.3, best_c * 0.6, best_c * 1.5, best_c * 3.0])
)
neighbors_c = [float(min(max(c, 1e-4), 200.0)) for c in neighbors_c]

if best_pen == "l1":
    param_grid_lr_r2 = [
        {"clf__C": neighbors_c, "clf__penalty": ["l1"], "clf__solver": ["saga"]}
    ]
else:
    param_grid_lr_r2 = [
        {"clf__C": neighbors_c, "clf__penalty": ["l2"], "clf__solver": [best_sol]}
    ]
grid_lr2 = GridSearchCV(
    pipe_lr, param_grid_lr_r2, cv=cv_splitter,
    scoring="f1_macro", n_jobs=-1, refit=True,
)
grid_lr2.fit(X_train, y_train)

lr_r2_gain = grid_lr2.best_score_ - grid_lr.best_score_
if lr_r2_gain < TUNING_PLATEAU_EPS:
    print(
        f"[逻辑回归] 第二轮相对第一轮 CV(Macro F1) 提升 {lr_r2_gain:.5f}，"
        "已较小；再扩大网格通常收益有限。"
    )
final_lr = grid_lr2 if grid_lr2.best_score_ >= grid_lr.best_score_ else grid_lr

y_pred_lr = final_lr.predict(X_test)
stab_lr = _stability_cv(final_lr.best_estimator_, X_train, y_train, cv_splitter)
row_lr = _collect_result(
    "Logistic Regression (Pipeline, Grid×2)",
    final_lr.best_params_,
    final_lr.best_score_,
    stab_lr,
    y_test,
    y_pred_lr,
    SEVERITY_ORDER,
)
results_rows.append(row_lr)
model_summaries["lr"] = row_lr
print("逻辑回归 best CV macro F1:", final_lr.best_score_)"""
    )
)

cells.append(
    cell_md("""## 3. 决策树（未标准化；多轮 GridSearch）""")
)

cells.append(
    cell_code(
        """dt = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
param_grid_dt_r1 = {
    "max_depth": [8, 12, 16, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "ccp_alpha": [0.0, 0.0001, 0.001],
    "max_leaf_nodes": [None, 64, 128],
}
grid_dt = GridSearchCV(
    dt, param_grid_dt_r1, cv=cv_splitter,
    scoring="f1_macro", n_jobs=-1, refit=True,
)
grid_dt.fit(X_train, y_train)

bp = grid_dt.best_params_
param_grid_dt_r2 = {
    "max_depth": _neighbors_none(bp["max_depth"], [8, 12, 16, 20, None]),
    "min_samples_split": _neighbors_int(bp["min_samples_split"], [2, 5, 10, 15]),
    "min_samples_leaf": _neighbors_int(bp["min_samples_leaf"], [1, 2, 4, 8]),
    "ccp_alpha": sorted(
        set([bp["ccp_alpha"], bp["ccp_alpha"] * 0.3, bp["ccp_alpha"] * 3])
    ),
    "max_leaf_nodes": [bp["max_leaf_nodes"]],
}
grid_dt2 = GridSearchCV(
    dt, param_grid_dt_r2, cv=cv_splitter,
    scoring="f1_macro", n_jobs=-1, refit=True,
)
grid_dt2.fit(X_train, y_train)

dt_r2_gain = grid_dt2.best_score_ - grid_dt.best_score_
if dt_r2_gain < TUNING_PLATEAU_EPS:
    print(
        f"[决策树] 第二轮相对第一轮 CV 提升 {dt_r2_gain:.5f}，继续加网格收益可能有限。"
    )

final_dt = grid_dt2 if grid_dt2.best_score_ >= grid_dt.best_score_ else grid_dt
if RUN_TREE_ROUND3:
    bp2 = final_dt.best_params_
    param_grid_dt_r3 = _refine_grid_decision_tree(bp2)
    cv_before_r3 = final_dt.best_score_
    grid_dt3 = GridSearchCV(
        dt, param_grid_dt_r3, cv=cv_splitter,
        scoring="f1_macro", n_jobs=-1, refit=True,
    )
    grid_dt3.fit(X_train, y_train)
    if grid_dt3.best_score_ > cv_before_r3:
        final_dt = grid_dt3
        print(
            f"[决策树] 第三轮 CV(Macro F1) 从 {cv_before_r3:.5f} 提升至 {grid_dt3.best_score_:.5f}。"
        )
    else:
        print(
            f"[决策树] 第三轮未超过 {cv_before_r3:.5f}，保留上一轮最优。"
        )

y_pred_dt = final_dt.predict(X_test)
stab_dt = _stability_cv(final_dt.best_estimator_, X_train, y_train, cv_splitter)
row_dt = _collect_result(
    "Decision Tree (Grid 多轮)",
    final_dt.best_params_,
    final_dt.best_score_,
    stab_dt,
    y_test,
    y_pred_dt,
    SEVERITY_ORDER,
)
results_rows.append(row_dt)
model_summaries["dt"] = row_dt
print("决策树 best CV macro F1:", final_dt.best_score_)"""
    )
)

cells.append(
    cell_md("""## 4. 随机森林（未标准化；多轮 GridSearch）""")
)

cells.append(
    cell_code(
        """rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
param_grid_rf_r1 = {
    "n_estimators": [100, 200, 300],
    "max_depth": [12, 18, 24, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.3, 0.5],
}
grid_rf = GridSearchCV(
    rf, param_grid_rf_r1, cv=cv_splitter,
    scoring="f1_macro", n_jobs=-1, refit=True,
)
grid_rf.fit(X_train, y_train)

bp_rf = grid_rf.best_params_
param_grid_rf_r2 = {
    "n_estimators": _neighbors_int(bp_rf["n_estimators"], [80, 120, 200, 250, 350]),
    "max_depth": _neighbors_none(bp_rf["max_depth"], [10, 15, 20, 24, None]),
    "min_samples_split": [bp_rf["min_samples_split"]],
    "min_samples_leaf": [bp_rf["min_samples_leaf"]],
    "max_features": [bp_rf["max_features"]],
}
grid_rf2 = GridSearchCV(
    rf, param_grid_rf_r2, cv=cv_splitter,
    scoring="f1_macro", n_jobs=-1, refit=True,
)
grid_rf2.fit(X_train, y_train)

rf_r2_gain = grid_rf2.best_score_ - grid_rf.best_score_
if rf_r2_gain < TUNING_PLATEAU_EPS:
    print(
        f"[随机森林] 第二轮相对第一轮 CV 提升 {rf_r2_gain:.5f}，再扩大搜索收益可能有限。"
    )

final_rf = grid_rf2 if grid_rf2.best_score_ >= grid_rf.best_score_ else grid_rf
if RUN_TREE_ROUND3:
    bp2_rf = final_rf.best_params_
    param_grid_rf_r3 = _refine_grid_random_forest(bp2_rf)
    cv_rf_before_r3 = final_rf.best_score_
    grid_rf3 = GridSearchCV(
        rf, param_grid_rf_r3, cv=cv_splitter,
        scoring="f1_macro", n_jobs=-1, refit=True,
    )
    grid_rf3.fit(X_train, y_train)
    if grid_rf3.best_score_ > cv_rf_before_r3:
        final_rf = grid_rf3
        print(
            f"[随机森林] 第三轮 CV(Macro F1) 从 {cv_rf_before_r3:.5f} 提升至 {grid_rf3.best_score_:.5f}。"
        )
    else:
        print(
            f"[随机森林] 第三轮未超过 {cv_rf_before_r3:.5f}，保留上一轮最优。"
        )

y_pred_rf = final_rf.predict(X_test)
stab_rf = _stability_cv(final_rf.best_estimator_, X_train, y_train, cv_splitter)
row_rf = _collect_result(
    "Random Forest (Grid 多轮)",
    final_rf.best_params_,
    final_rf.best_score_,
    stab_rf,
    y_test,
    y_pred_rf,
    SEVERITY_ORDER,
)
results_rows.append(row_rf)
model_summaries["rf"] = row_rf
print("随机森林 best CV macro F1:", final_rf.best_score_)"""
    )
)

cells.append(
    cell_md("""## 5. XGBoost（对照 benchmark，未调 GridSearch）""")
)

cells.append(
    cell_code(
        """row_xgb = None
y_pred_xgb = None
le = None
_xgb_kw = None

if XGBClassifier is None:
    print("未安装 xgboost，跳过 XGBoost。")
else:
    le = LabelEncoder()
    le.fit(SEVERITY_ORDER)
    y_train_xgb = le.transform(y_train.astype(str))
    _xgb_kw = dict(
        random_state=RANDOM_STATE,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    try:
        xgb = XGBClassifier(**_xgb_kw, use_label_encoder=False)
    except TypeError:
        xgb = XGBClassifier(**_xgb_kw)
    xgb.fit(X_train, y_train_xgb)
    y_pred_xgb_idx = xgb.predict(X_test)
    y_pred_xgb = le.inverse_transform(y_pred_xgb_idx.astype(int))
    y_pred_xgb = pd.Categorical(
        y_pred_xgb, categories=SEVERITY_ORDER, ordered=True
    )
    stab_xgb = _stability_cv_xgb(xgb, X_train, y_train_xgb, cv_splitter)
    row_xgb = _collect_result(
        "XGBoost (benchmark, 未 GridSearch)",
        {"note": "固定较强默认超参"},
        stab_xgb["cv_mean_macro_f1"],
        stab_xgb,
        y_test,
        y_pred_xgb,
        SEVERITY_ORDER,
    )
    results_rows.append(row_xgb)
    model_summaries["xgb"] = row_xgb

    print("XGBoost 已训练（CV 稳定性见汇总表）。")"""
    )
)

cells.append(
    cell_md("""## 6. 结果汇总表（DataFrame）""")
)

cells.append(
    cell_code(
        """summary_df = pd.DataFrame(results_rows)
print("\n========== 各模型结果汇总 ==========")
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 200)
summary_df  # 在 Jupyter 中会显示表格"""
    )
)

cells.append(
    cell_md("""## 7. 重复分层划分稳定性（测试 Macro F1 的 mean / std）""")
)

cells.append(
    cell_code(
        """if RUN_REPEATED_HOLDOUT:
    print(
        f"========== 重复分层划分稳定性（{N_REPEATED_HOLDOUT} 次）=========="
    )
    for label, est in [
        ("LogisticRegression", final_lr.best_estimator_),
        ("DecisionTree", final_dt.best_estimator_),
        ("RandomForest", final_rf.best_estimator_),
    ]:
        mu, sigma = _repeated_holdout_macro_f1(
            est, X, y, n_times=N_REPEATED_HOLDOUT, test_size=TEST_SIZE
        )
        print(f"  {label}: mean={mu:.5f}, std={sigma:.5f}")
    if row_xgb is not None and le is not None and _xgb_kw is not None:
        mu, sigma = _repeated_holdout_macro_f1_xgb(
            X, y, le, n_times=N_REPEATED_HOLDOUT,
            test_size=TEST_SIZE, xgb_params=_xgb_kw,
        )
        print(f"  XGBoost: mean={mu:.5f}, std={sigma:.5f}")
else:
    print("已关闭 RUN_REPEATED_HOLDOUT，跳过本步。")"""
    )
)

cells.append(
    cell_md("""## 8. 混淆矩阵与 classification_report（每类 precision/recall/f1）""")
)

cells.append(
    cell_code(
        """for name, pred in [
    ("LogisticRegression", y_pred_lr),
    ("DecisionTree", y_pred_dt),
    ("RandomForest", y_pred_rf),
]:
    print(f"\\n----- {name} -----")
    print(
        "混淆矩阵（行=真实，列=预测）:\\n",
        confusion_matrix(y_test, pred, labels=SEVERITY_ORDER),
    )
    print(
        classification_report(
            y_test, pred, labels=SEVERITY_ORDER, zero_division=0
        )
    )

if row_xgb is not None and y_pred_xgb is not None:
    print("\\n----- XGBoost -----")
    print(
        confusion_matrix(y_test, y_pred_xgb, labels=SEVERITY_ORDER)
    )
    print(
        classification_report(
            y_test, y_pred_xgb, labels=SEVERITY_ORDER, zero_division=0
        )
    )"""
    )
)

cells.append(
    cell_md("""## 9. 模型选择建议（预测最优 vs 教学解释）""")
)

cells.append(
    cell_code(
        """print("========== 模型选择建议 ==========")
candidates = [
    ("Logistic Regression", row_lr["test_macro_f1"], row_lr),
    ("Decision Tree", row_dt["test_macro_f1"], row_dt),
    ("Random Forest", row_rf["test_macro_f1"], row_rf),
]
if row_xgb is not None:
    candidates.append(("XGBoost", row_xgb["test_macro_f1"], row_xgb))

pred_best = max(candidates, key=lambda x: x[1])
print(
    f"【Predictive best】测试 Macro F1: {pred_best[0]}, F1={pred_best[1]:.5f}"
)
if pred_best[0] == "XGBoost":
    print("（XGBoost 常为强 benchmark；作业中可如实报告。）")

tree_rows = [("Decision Tree", row_dt), ("Random Forest", row_rf)]
edu_name = max(
    tree_rows,
    key=lambda t: (t[1]["test_macro_f1"], -t[1]["cv_std_macro_f1"]),
)[0]
print(
    f"【Educationally preferable】推荐: {edu_name} 或决策树（规则直观）；"
    "随机森林可看特征重要性。"
)
print(
    "最优模型不一定只看分数最高：分数接近时可选更稳定（CV std 更小）、更易讲解的模型。"
)
print(
    f"若多轮 GridSearch 提升已低于约 {TUNING_PLATEAU_EPS}，可写明「继续扩大网格收益递减」。"
)"""
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_name = "model.ipynb"
try:
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
except PermissionError:
    out_name = "model_workflow.ipynb"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("注意: model.ipynb 被占用，已写入", out_name, "请关闭占用后重命名或复制。")

print("Wrote", out_name, "with", len(cells), "cells")

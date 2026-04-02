# -*- coding: utf-8 -*-
"""
肥胖等级多分类建模脚本（课程作业汇报版）

数据：Obesity_Data_clean_onehot.xlsx
- 已含二值编码与 one-hot（虚拟变量），本脚本仅做列类型检查与 bool→int，不重复编码。

流程概要：读入数据 → 划分特征/标签 → 分层 train/test →
逻辑回归（Pipeline：StandardScaler+LR，GridSearchCV，无泄漏）→
决策树 / 随机森林（多轮 GridSearch，原始特征）→
XGBoost（对照 benchmark）→ 多指标评价、混淆矩阵、稳定性、模型选择建议。
"""

import warnings

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
    XGBClassifier = None

# ---------------------------------------------------------------------------
# 【以下为配置常量；import 以外逻辑均配有中文注释】
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
# 是否做「多次不同 random_state 的 train/test」以观察测试 Macro F1 波动（与 K 折 CV 互补）
RUN_REPEATED_HOLDOUT = True
N_REPEATED_HOLDOUT = 3
# 第三轮树模型细化：在第二轮最优附近再搜一小步；若设为 False 可略省时间
RUN_TREE_ROUND3 = True
# 若第二轮相对第一轮 CV 提升小于该阈值，认为「再搜收益有限」，仍可做第三轮微调但会打印提示
TUNING_PLATEAU_EPS = 0.0008

# 肥胖等级有序类别（从最瘦到最胖），与数据/报告顺序一致
SEVERITY_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]


def main():
    # ------------------------------------------------------------------
    # 1. 读取数据：建模输入表已完成编码，此处不做 one-hot
    # ------------------------------------------------------------------
    path = "Obesity_Data_clean_onehot.xlsx"
    df = pd.read_excel(path)
    print("数据集形状:", df.shape)
    # 作业要求：先打印列名，若与预期不符可据此改列名或特征列表
    print("列名列表:", list(df.columns))

    # ------------------------------------------------------------------
    # 2. 目标变量：转为有序分类，便于 classification_report / 混淆矩阵顺序一致
    # ------------------------------------------------------------------
    if "obesity_level" not in df.columns:
        raise ValueError("缺少目标列 obesity_level，请检查 Excel 列名。")
    df["obesity_level"] = pd.Categorical(
        df["obesity_level"], categories=SEVERITY_ORDER, ordered=True
    )

    # ------------------------------------------------------------------
    # 3. 划分特征矩阵 X 与标签向量 y（其余列均为已编码特征）
    # ------------------------------------------------------------------
    X = df.drop(columns=["obesity_level"])
    y = df["obesity_level"]
    # 将布尔型虚拟变量转为 0/1，避免部分 sklearn 版本对 bool 处理不一致
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(np.int8)

    # ------------------------------------------------------------------
    # 4. 训练集 / 测试集划分：stratify=y 保证 7 类比例与总体一致，减轻偶然划分偏差
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("训练集:", X_train.shape, "测试集:", X_test.shape)

    # ------------------------------------------------------------------
    # 评价指标（作业约定）：
    # - 主指标 Macro F1：每类 F1 算术平均，类别略不平衡时比 Accuracy 更公平，用于 GridSearch 与选最优模型。
    # - Accuracy：整体正确率；大类主导时可能掩盖小类错误，不能单独作为唯一标准。
    # - Weighted F1：按样本数加权，更贴近「整体加权」表现。
    # - Precision（宏平均）：各类查准率平均，关注「预测为某类时有多准」。
    # - Recall（宏平均）：各类查全率平均，关注「某类样本被找出多少」。
    # - 混淆矩阵：展示类间错分模式，便于讲解与改进特征。
    # 选参与最终选型：以 Macro F1 为主，结合上述指标、CV 稳定性与可解释性。
    # ------------------------------------------------------------------

    # 5 折分层交叉验证划分器：与 train_test_split 配合，用于 GridSearch 与稳定性 std
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results_rows = []
    model_summaries = {}

    # ==================================================================
    # 5. 逻辑回归：Pipeline(StandardScaler → LogisticRegression)
    #    - 为何标准化：系数与正则惩罚对各特征尺度敏感；使优化与 L1/L2 惩罚可比。
    #    - 为何用 Pipeline + GridSearchCV：每一折只在训练折 fit StandardScaler，再训 LR，
    #      避免「先用全训练集标准化再划分 CV」导致测试折信息泄漏。
    #    - 树模型基于划分、不依赖线性尺度，故 DT/RF/XGB 使用未标准化但已编码的 X。
    # ==================================================================
    pipe_lr = Pipeline(
        [
            # StandardScaler：对训练折估计均值方差，同一折验证集用 transform
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    # sklearn>=1.5 已移除 multi_class 参数，多分类由数据与 solver 自动处理
                ),
            ),
        ]
    )
    # 参数含义：C 越小正则越强；penalty l1/l2；solver 必须与 penalty 合法组合（l1 多用 saga）
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
        pipe_lr,
        param_grid_lr_r1,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    # 传入原始 X_train；Pipeline 在 CV 内部逐折 fit scaler，无泄漏
    grid_lr.fit(X_train, y_train)

    # 第二轮：在最优 C 邻域细化（多轮尝试、控制规模）
    best_c = grid_lr.best_params_.get("clf__C", 1.0)
    best_pen = grid_lr.best_params_.get("clf__penalty", "l2")
    best_sol = grid_lr.best_params_.get("clf__solver", "lbfgs")
    neighbors_c = sorted(
        set(
            [
                best_c,
                best_c * 0.3,
                best_c * 0.6,
                best_c * 1.5,
                best_c * 3.0,
            ]
        )
    )
    neighbors_c = [float(min(max(c, 1e-4), 200.0)) for c in neighbors_c]

    if best_pen == "l1":
        param_grid_lr_r2 = [
            {"clf__C": neighbors_c, "clf__penalty": ["l1"], "clf__solver": ["saga"]}
        ]
    else:
        param_grid_lr_r2 = [
            {
                "clf__C": neighbors_c,
                "clf__penalty": ["l2"],
                "clf__solver": [best_sol],
            }
        ]
    grid_lr2 = GridSearchCV(
        pipe_lr,
        param_grid_lr_r2,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    grid_lr2.fit(X_train, y_train)

    lr_r2_gain = grid_lr2.best_score_ - grid_lr.best_score_
    if lr_r2_gain < TUNING_PLATEAU_EPS:
        print(
            f"[逻辑回归] 第二轮相对第一轮 CV(Macro F1) 提升 {lr_r2_gain:.5f}，"
            "已较小；再扩大网格通常收益有限（可在报告中说明）。"
        )
    final_lr = grid_lr2 if grid_lr2.best_score_ >= grid_lr.best_score_ else grid_lr

    # 在固定 test 集上预测；最优参数来自 GridSearch 的 refit 全训练集模型
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

    # ==================================================================
    # 6. 决策树：未标准化；多轮 GridSearch
    #    max_depth：限制树深，控制过拟合；min_samples_split：分裂前节点最少样本；
    #    min_samples_leaf：叶节点最少样本；ccp_alpha：代价复杂度剪枝；
    #    max_leaf_nodes：叶数上限，限制模型容量。
    # ==================================================================
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    param_grid_dt_r1 = {
        "max_depth": [8, 12, 16, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "ccp_alpha": [0.0, 0.0001, 0.001],
        "max_leaf_nodes": [None, 64, 128],
    }
    grid_dt = GridSearchCV(
        dt,
        param_grid_dt_r1,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
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
        dt,
        param_grid_dt_r2,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    grid_dt2.fit(X_train, y_train)

    dt_r2_gain = grid_dt2.best_score_ - grid_dt.best_score_
    if dt_r2_gain < TUNING_PLATEAU_EPS:
        print(
            f"[决策树] 第二轮相对第一轮 CV 提升 {dt_r2_gain:.5f}，"
            "继续加网格收益可能有限；下面第三轮为可选细化。"
        )

    # 第三轮（可选）：在最优参数附近再缩小一步，侧重教学用树的可解释与泛化平衡
    final_dt = grid_dt2 if grid_dt2.best_score_ >= grid_dt.best_score_ else grid_dt
    if RUN_TREE_ROUND3:
        bp2 = final_dt.best_params_
        param_grid_dt_r3 = _refine_grid_decision_tree(bp2)
        cv_before_r3 = final_dt.best_score_
        grid_dt3 = GridSearchCV(
            dt,
            param_grid_dt_r3,
            cv=cv_splitter,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        grid_dt3.fit(X_train, y_train)
        # 仅当第三轮 CV(Macro F1) 严格优于进入第三轮前的模型时才替换（避免噪声略降仍替换）
        if grid_dt3.best_score_ > cv_before_r3:
            final_dt = grid_dt3
            print(
                f"[决策树] 第三轮 GridSearch 使 CV(Macro F1) 从 {cv_before_r3:.5f} 提升至 {grid_dt3.best_score_:.5f}。"
            )
        else:
            print(
                f"[决策树] 第三轮未超过进入第三轮前的 CV({cv_before_r3:.5f})，保留上一轮最优参数。"
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

    # ==================================================================
    # 7. 随机森林：未标准化；n_estimators 树棵数；max_features 每树可用特征比例/规则
    # ==================================================================
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    param_grid_rf_r1 = {
        "n_estimators": [100, 200, 300],
        "max_depth": [12, 18, 24, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.3, 0.5],
    }
    grid_rf = GridSearchCV(
        rf,
        param_grid_rf_r1,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
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
        rf,
        param_grid_rf_r2,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    grid_rf2.fit(X_train, y_train)

    rf_r2_gain = grid_rf2.best_score_ - grid_rf.best_score_
    if rf_r2_gain < TUNING_PLATEAU_EPS:
        print(
            f"[随机森林] 第二轮相对第一轮 CV 提升 {rf_r2_gain:.5f}，"
            "再扩大搜索收益可能有限；第三轮为邻域微调。"
        )

    final_rf = grid_rf2 if grid_rf2.best_score_ >= grid_rf.best_score_ else grid_rf
    if RUN_TREE_ROUND3:
        bp2_rf = final_rf.best_params_
        param_grid_rf_r3 = _refine_grid_random_forest(bp2_rf)
        cv_rf_before_r3 = final_rf.best_score_
        grid_rf3 = GridSearchCV(
            rf,
            param_grid_rf_r3,
            cv=cv_splitter,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        grid_rf3.fit(X_train, y_train)
        if grid_rf3.best_score_ > cv_rf_before_r3:
            final_rf = grid_rf3
            print(
                f"[随机森林] 第三轮 GridSearch 使 CV(Macro F1) 从 {cv_rf_before_r3:.5f} 提升至 {grid_rf3.best_score_:.5f}。"
            )
        else:
            print(
                f"[随机森林] 第三轮未超过进入第三轮前的 CV({cv_rf_before_r3:.5f})，保留上一轮最优参数。"
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

    # ==================================================================
    # 8. XGBoost：对照 benchmark，不做 GridSearch；标签整数编码，特征不标准化
    # ==================================================================
    row_xgb = None
    y_pred_xgb = None
    le = None
    _xgb_kw = None
    if XGBClassifier is None:
        print("未安装 xgboost，跳过 XGBoost 对照实验。")
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
        # 模型训练：在完整训练集上拟合（与前面树模型一致）；评估在独立测试集
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

    # ------------------------------------------------------------------
    # 9. 汇总表 DataFrame：便于作业对比（含 best_params、best CV、测试集各指标、稳定性）
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(results_rows)
    print("\n========== 各模型结果汇总（主指标：Macro F1；含 Accuracy / Precision / Recall / F1）==========")
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", 200)
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 9b. 稳定性：重复分层 holdout，看测试 Macro F1 的 mean / std（补充单次划分偶然性）
    # ------------------------------------------------------------------
    if RUN_REPEATED_HOLDOUT:
        print(
            f"\n========== 重复分层划分稳定性（{N_REPEATED_HOLDOUT} 次，指标：测试 Macro F1）=========="
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
                X,
                y,
                le,
                n_times=N_REPEATED_HOLDOUT,
                test_size=TEST_SIZE,
                xgb_params=_xgb_kw,
            )
            print(f"  XGBoost: mean={mu:.5f}, std={sigma:.5f}")

    # ------------------------------------------------------------------
    # 10. 模型评估：混淆矩阵 + classification_report（含每类 precision/recall/f1）
    # ------------------------------------------------------------------
    for name, pred in [
        ("LogisticRegression", y_pred_lr),
        ("DecisionTree", y_pred_dt),
        ("RandomForest", y_pred_rf),
    ]:
        print(f"\n----- {name}：Confusion Matrix & classification report -----")
        cm = confusion_matrix(y_test, pred, labels=SEVERITY_ORDER)
        print("混淆矩阵（行=真实类别，列=预测类别）:\n", cm)
        print(
            classification_report(
                y_test, pred, labels=SEVERITY_ORDER, zero_division=0
            )
        )

    if row_xgb is not None and y_pred_xgb is not None:
        print("\n----- XGBoost：Confusion Matrix & classification report -----")
        print(
            "混淆矩阵（行=真实，列=预测）:\n",
            confusion_matrix(y_test, y_pred_xgb, labels=SEVERITY_ORDER),
        )
        print(
            classification_report(
                y_test, y_pred_xgb, labels=SEVERITY_ORDER, zero_division=0
            )
        )

    # ------------------------------------------------------------------
    # 11. 模型选择建议
    # - 最优模型不一定只是测试分数最高：若分数接近，可优先更稳定（CV std 小）、更易讲解的模型。
    # - 树模型易解释（规则、特征重要性），适合 STEM 课堂演示「数据→决策」；随机森林在性能与可解释性间常更平衡。
    # - XGBoost 若最强应如实写出，并说明其作 benchmark；教学仍可选用树模型讲解释。
    # ------------------------------------------------------------------
    print("\n========== 模型选择建议（作业汇报）==========")
    candidates = [
        ("Logistic Regression", row_lr["test_macro_f1"], row_lr),
        ("Decision Tree", row_dt["test_macro_f1"], row_dt),
        ("Random Forest", row_rf["test_macro_f1"], row_rf),
    ]
    if row_xgb is not None:
        candidates.append(("XGBoost", row_xgb["test_macro_f1"], row_xgb))

    pred_best = max(candidates, key=lambda x: x[1])
    print(
        f"【Predictive best model】（以当前测试集 Macro F1 为主）: {pred_best[0]}, "
        f"Macro F1 = {pred_best[1]:.5f}。"
    )
    if pred_best[0] == "XGBoost":
        print(
            "  （说明：XGBoost 为梯度提升树集成，在本数据上常较强；作业中应如实报告，并保留作性能上限参考。）"
        )

    # 教学优先：在调参树模型中选测试 Macro F1 较高且 CV 相对稳定者
    tree_rows = [("Decision Tree", row_dt), ("Random Forest", row_rf)]
    edu_name = max(
        tree_rows,
        key=lambda t: (t[1]["test_macro_f1"], -t[1]["cv_std_macro_f1"]),
    )[0]
    print(
        f"【Educationally preferable model】（便于向学生解释）: 推荐 {edu_name} 或决策树（单棵树规则最直观）；"
        "随机森林可配合特征重要性条形图。逻辑回归可展示标准化后系数符号与风险因子讨论。"
    )
    print(
        "【为何树模型更易解释】单棵树对应一组 if-then 规则；随机森林可汇总特征重要性；"
        "学生易将「特征→分裂→类别」与真实决策联系。"
    )
    print(
        "【STEM lesson 展示】可用单棵决策树可视化、或森林的 Top-K 特征；对比黑箱时说明 XGBoost 仍依赖复杂集成。"
    )
    print(
        "【调参收益】若多轮 GridSearch 后 CV 或测试 Macro F1 提升已低于约 "
        f"{TUNING_PLATEAU_EPS}，可在报告中写明「继续扩大网格收益递减」，避免过拟合与运行时间过长。"
    )

    return summary_df, model_summaries


def _refine_grid_decision_tree(bp):
    """第三轮决策树：在最优参数附近极小网格，控制组合数。"""
    md = bp["max_depth"]
    if md is None:
        depth_opts = [16, 20, None]
    else:
        depth_opts = sorted(
            set(
                [max(4, md - 2), md - 1, md, md + 1, min(md + 2, 30)]
            )
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
    """第三轮随机森林：微调 n_estimators 与 max_depth，其余固定以控制计算量。"""
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
    # 对当前最优估计器在训练集上做 K 折交叉验证，返回 Macro F1 的均值与标准差
    scores = cross_val_score(
        estimator,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
    )
    return {"cv_mean_macro_f1": float(np.mean(scores)), "cv_std_macro_f1": float(np.std(scores))}


def _stability_cv_xgb(xgb_model, X_train, y_train_int, cv_splitter):
    scores = cross_val_score(
        xgb_model,
        X_train,
        y_train_int,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
    )
    return {"cv_mean_macro_f1": float(np.mean(scores)), "cv_std_macro_f1": float(np.std(scores))}


def _repeated_holdout_macro_f1(estimator, X, y, n_times, test_size):
    scores = []
    for k in range(n_times):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
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
            X,
            y,
            test_size=test_size,
            stratify=y,
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


def _collect_result(
    model_name,
    best_params,
    best_cv_score,
    stability,
    y_test,
    y_pred,
    label_order,
):
    # 测试集：Accuracy；宏平均与加权平均的 Precision / Recall / F1；混淆矩阵用于附录与课堂展示
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
    """在 max_depth 为整数时取邻域；None 表示不限制深度时单独处理。"""
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
    """在整数超参（如 min_samples_split）附近取候选，避免网格爆炸。"""
    out = {int(val)}
    for p in pool:
        if abs(p - val) <= max(3, val // 2):
            out.add(p)
    return sorted(out)


if __name__ == "__main__":
    main()

"""
Telco Customer Churn — End-to-End SQL + Explainable ML + Financial ROI Pipeline
Author: Mostafa Saad
Dataset: IBM Telco Customer Churn (7,043 real customer records)
"""

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DB_PATH = ROOT / "data" / "telco_churn.db"
SQL_PATH = ROOT / "sql" / "01_schema_and_views.sql"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# AMOLED Visual Theme
BG = "#000000"
CARD = "#0A0C12"
CYAN = "#00E5FF"
VIOLET = "#A78BFA"
EMERALD = "#10B981"
AMBER = "#F59E0B"
ROSE = "#F43F5E"
TEXT = "#FFFFFF"
MUTED = "#94A3B8"
GRID = "#1E2433"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD,
    "axes.edgecolor": "#262D3F",
    "axes.labelcolor": MUTED,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.family": "DejaVu Sans",
})


def ingest_and_run_sql() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    df_raw = pd.read_csv(RAW_CSV, dtype=str)
    conn = sqlite3.connect(DB_PATH)
    df_raw.to_sql("raw_telco_churn", conn, if_exists="replace", index=False)
    conn.executescript(SQL_PATH.read_text(encoding="utf-8"))

    qa = pd.read_sql_query("SELECT * FROM vw_churn_qa", conn).iloc[0].to_dict()
    df = pd.read_sql_query("SELECT * FROM vw_churn_features", conn)
    seg = pd.read_sql_query("SELECT * FROM vw_segment_summary", conn)
    conn.close()
    return df, qa, seg


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_prob)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_t[idx])) < 2:
            continue
        scores.append(metric_fn(y_t[idx], y_p[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_retention_roi(y_true, y_prob, monthly_charges, threshold, offer_cost=100.0, save_rate=0.35):
    """
    Realistic Telco retention economics:
    - Targeting a customer costs `offer_cost` ($100 retention incentive/outreach).
    - If they were truly going to churn, intervention saves them with probability `save_rate` (35%),
      preserving 12 months of MonthlyCharges (minus offer_cost).
    - False positives still incur `offer_cost` with zero incremental LTV gain.
    """
    pred_pos = y_prob >= threshold
    tp_mask = pred_pos & (y_true == 1)
    fp_mask = pred_pos & (y_true == 0)
    saved_value = np.sum(monthly_charges[tp_mask] * 12.0 * save_rate)
    total_cost = np.sum(pred_pos) * offer_cost
    net_profit = float(saved_value - total_cost)
    return {
        "threshold": float(threshold),
        "targeted": int(np.sum(pred_pos)),
        "true_churners_caught": int(np.sum(tp_mask)),
        "false_positives": int(np.sum(fp_mask)),
        "campaign_cost_usd": round(float(total_cost), 2),
        "expected_saved_rev_usd": round(float(saved_value), 2),
        "net_profit_usd": round(net_profit, 2),
        "roi_pct": round(100.0 * net_profit / total_cost, 1) if total_cost > 0 else 0.0,
    }


def main():
    print("[1/5] Ingesting IBM Telco dataset into SQLite and running SQL views...")
    df, qa, seg = ingest_and_run_sql()
    print("QA Summary:", qa)

    # Engineered Python ratio features (safe against zero division)
    df["avg_monthly_implied"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
    df["price_hike_ratio"] = np.where(df["avg_monthly_implied"] > 0, df["MonthlyCharges"] / df["avg_monthly_implied"], 1.0)
    df["charge_per_service"] = df["MonthlyCharges"] / np.maximum(1, df["active_services_count"])
    df["has_security_or_support"] = ((df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")).astype(int)

    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "contract_months",
        "active_services_count",
        "charge_per_service",
        "price_hike_ratio",
        "SeniorCitizen",
        "is_new_customer",
        "total_charges_was_missing",
        "is_electronic_check",
        "is_autopay",
        "has_security_or_support",
    ]
    cat_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    X = df[num_cols + cat_cols].copy()
    y = df["churn_flag"].values
    meta = df[["customerID", "Contract", "tenure_cohort", "SeniorCitizen", "MonthlyCharges", "annual_revenue_12m"]].copy()

    # Stratified 60 / 20 / 20 Train / Val / Test split (zero leakage)
    X_trainval, X_test, y_trainval, y_test, meta_trainval, meta_test = train_test_split(
        X, y, meta, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_trainval, y_trainval, meta_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )
    print(f"Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    models = {
        "Naive Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=12, random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=220, learning_rate=0.04, num_leaves=15, max_depth=4,
            min_child_samples=30, subsample=0.85, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0, random_state=42, verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=220, learning_rate=0.04, max_depth=3, min_child_weight=8,
            subsample=0.85, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
    }

    print("[2/5] Training & benchmarking 5 models with 5-fold Stratified CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    val_probs = {}
    test_probs = {}
    fitted_pipes = {}

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        p_val = pipe.predict_proba(X_val)[:, 1]
        p_test = pipe.predict_proba(X_test)[:, 1]

        val_probs[name] = p_val
        test_probs[name] = p_test
        fitted_pipes[name] = pipe

        auc_test = roc_auc_score(y_test, p_test)
        pr_test = average_precision_score(y_test, p_test)
        brier_test = brier_score_loss(y_test, p_test)
        pred_50 = (p_test >= 0.50).astype(int)
        ci_low, ci_high = bootstrap_ci(y_test, p_test, roc_auc_score) if name != "Naive Baseline" else (0.5, 0.5)

        results[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "val_roc_auc": round(float(roc_auc_score(y_val, p_val)), 4),
            "test_roc_auc": round(float(auc_test), 4),
            "test_roc_auc_ci95": [round(ci_low, 4), round(ci_high, 4)],
            "test_pr_auc": round(float(pr_test), 4),
            "test_brier": round(float(brier_test), 4),
            "test_f1_at_50": round(float(f1_score(y_test, pred_50, zero_division=0)), 4),
            "test_precision_at_50": round(float(precision_score(y_test, pred_50, zero_division=0)), 4),
            "test_recall_at_50": round(float(recall_score(y_test, pred_50, zero_division=0)), 4),
        }
        print(f"  {name:20s} | CV AUC={results[name]['cv_roc_auc_mean']:.4f} | Test AUC={auc_test:.4f} | PR-AUC={pr_test:.4f} | Brier={brier_test:.4f}")

    # Select best model by Validation ROC-AUC (strictly avoiding test-set leakage!)
    ml_names = [k for k in results if k != "Naive Baseline"]
    best_name = max(ml_names, key=lambda k: results[k]["val_roc_auc"])
    print(f"Selected Best Model (on Validation AUC): {best_name}")

    # Optimize financial threshold STRICTLY on validation set, then evaluate on held-out Test set
    thresholds = np.linspace(0.08, 0.85, 78)
    val_rois = [compute_retention_roi(y_val, val_probs[best_name], meta_val["MonthlyCharges"].values, t) for t in thresholds]
    best_t_val = max(val_rois, key=lambda r: r["net_profit_usd"])["threshold"]

    roi_default_test = compute_retention_roi(y_test, test_probs[best_name], meta_test["MonthlyCharges"].values, 0.50)
    roi_opt_test = compute_retention_roi(y_test, test_probs[best_name], meta_test["MonthlyCharges"].values, best_t_val)
    print(f"Optimal Threshold (from Val): {best_t_val:.2f}")
    print(f"Test Net Profit @0.50: ${roi_default_test['net_profit_usd']:,.2f} -> @{best_t_val:.2f}: ${roi_opt_test['net_profit_usd']:,.2f}")

    # Slice-based Error Analysis on held-out Test Set (per ml-best-practices)
    test_eval = meta_test.copy()
    test_eval["y_true"] = y_test
    test_eval["y_prob"] = test_probs[best_name]
    test_eval["y_pred_opt"] = (test_eval["y_prob"] >= best_t_val).astype(int)

    slice_audit = []
    for col in ["Contract", "tenure_cohort"]:
        for val, grp in test_eval.groupby(col):
            if len(grp) < 15 or grp["y_true"].nunique() < 2:
                continue
            slice_audit.append({
                "slice_col": col,
                "slice_val": str(val),
                "n": int(len(grp)),
                "actual_churn_pct": round(100.0 * grp["y_true"].mean(), 1),
                "roc_auc": round(float(roc_auc_score(grp["y_true"], grp["y_prob"])), 3),
                "recall_opt": round(float(recall_score(grp["y_true"], grp["y_pred_opt"], zero_division=0)), 3),
                "precision_opt": round(float(precision_score(grp["y_true"], grp["y_pred_opt"], zero_division=0)), 3),
            })

    # [3/5] SHAP Explainability on Best Tree Model (or XGBoost if LR won, compute SHAP cleanly!)
    print("[3/5] Computing SHAP explainability values...")
    best_pipe = fitted_pipes[best_name]
    prep_fitted = best_pipe.named_steps["prep"]
    clf_fitted = best_pipe.named_steps["clf"]
    feature_names = prep_fitted.get_feature_names_out()
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    X_test_trans = prep_fitted.transform(X_test)
    X_train_trans = prep_fitted.transform(X_train)

    if isinstance(clf_fitted, LogisticRegression):
        explainer = shap.LinearExplainer(clf_fitted, X_train_trans)
        shap_vals = explainer.shap_values(X_test_trans)
    else:
        explainer = shap.TreeExplainer(clf_fitted)
        raw_sv = explainer.shap_values(X_test_trans)
        shap_vals = raw_sv[1] if isinstance(raw_sv, list) else (raw_sv[:, :, 1] if raw_sv.ndim == 3 else raw_sv)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:12]
    top_shap_features = [
        {"feature": clean_names[i], "mean_abs_shap": round(float(mean_abs_shap[i]), 4)}
        for i in top_idx
    ]

    # Export Top High-Risk Actionable Retention List with individual top-3 SHAP reasons
    top_risk = test_eval.sort_values("y_prob", ascending=False).head(50).copy()
    reasons = []
    for row_i in top_risk.index:
        pos = test_eval.index.get_loc(row_i)
        sv_row = shap_vals[pos]
        top3 = np.argsort(sv_row)[::-1][:3]
        reasons.append(" | ".join([f"{clean_names[j]} (+{sv_row[j]:.2f})" for j in top3]))
    top_risk["top_3_churn_drivers"] = reasons
    top_risk.to_csv(OUT_DIR / "high_risk_action_list.csv", index=False)

    # Save metrics JSON
    metrics_payload = {
        "dataset": "IBM Telco Customer Churn (Real Public Data)",
        "qa_summary": qa,
        "splits": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "models": results,
        "selected_model": best_name,
        "optimal_threshold_val": round(float(best_t_val), 3),
        "test_roi_at_50": roi_default_test,
        "test_roi_at_optimal": roi_opt_test,
        "top_shap_features": top_shap_features,
        "slice_error_audit": slice_audit,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # [4/5] Generate 4 Publication-Grade AMOLED Figures
    print("[4/5] Rendering 4 AMOLED executive visualizations...")

    # --- Figure 1: EDA Churn Drivers ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=160)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Telco Churn EDA — Contract Lock-In & Early-Tenure Risk Cliff", color=TEXT, fontsize=15, fontweight="bold", y=0.97)

    c_order = ["Month-to-month", "One year", "Two year"]
    c_rates = [100 * df[df["Contract"] == c]["churn_flag"].mean() for c in c_order]
    c_cols = [ROSE, AMBER, EMERALD]
    bars = ax1.bar(c_order, c_rates, color=c_cols, width=0.55, edgecolor="#FFFFFF", lw=0.6)
    for b, r in zip(bars, c_rates):
        ax1.text(b.get_x() + b.get_width() / 2, r + 1.3, f"{r:.1f}%", ha="center", color=TEXT, fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 52)
    ax1.set_ylabel("Churn Rate (%)")
    ax1.set_title("Churn Rate by Contract Type", color=CYAN, fontweight="bold", pad=10)
    ax1.grid(axis="y", ls=":", alpha=0.6)

    t_order = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]
    t_rates = [100 * df[df["tenure_cohort"] == t]["churn_flag"].mean() for t in t_order]
    ax2.plot(t_order, t_rates, marker="o", color=CYAN, lw=2.6, markersize=8, markerfacecolor=BG, markeredgewidth=2.2)
    ax2.fill_between(range(len(t_order)), t_rates, color=CYAN, alpha=0.14)
    for i, r in enumerate(t_rates):
        ax2.text(i, r + 2.2, f"{r:.1f}%", ha="center", color=TEXT, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 64)
    ax2.set_ylabel("Churn Rate (%)")
    ax2.set_title("Churn Cliff Across Customer Tenure Cohorts", color=VIOLET, fontweight="bold", pad=10)
    ax2.grid(ls=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(FIG_DIR / "01_churn_eda_drivers.png", facecolor=BG)
    plt.close(fig)

    # --- Figure 2: Model Benchmark (ROC + PR + Calibration) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.9), dpi=160)
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Model Benchmark & Probability Calibration (Selected: {best_name})", color=TEXT, fontsize=14.5, fontweight="bold", y=0.97)
    palette = {"Logistic Regression": CYAN, "Random Forest": VIOLET, "LightGBM": EMERALD, "XGBoost": AMBER}

    for m_name, col in palette.items():
        p = test_probs[m_name]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax1.plot(fpr, tpr, color=col, lw=2.1, label=f"{m_name} ({results[m_name]['test_roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax2.plot(rec, prec, color=col, lw=2.1, label=f"{m_name} ({results[m_name]['test_pr_auc']:.3f})")
        prob_true, prob_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
        ax3.plot(prob_pred, prob_true, marker="o", color=col, lw=1.9, markersize=4.5, label=f"{m_name} (Brier {results[m_name]['test_brier']:.3f})")

    ax1.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax1.set_title("ROC Curve (Test Set)", color=CYAN, fontweight="bold")
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.grid(ls=":", alpha=0.5); ax1.legend(frameon=False, fontsize=8, loc="lower right")

    ax2.axhline(y_test.mean(), color=MUTED, ls="--", lw=1, label=f"Baseline ({y_test.mean():.2f})")
    ax2.set_title("Precision-Recall Curve", color=VIOLET, fontweight="bold")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.grid(ls=":", alpha=0.5); ax2.legend(frameon=False, fontsize=8, loc="lower left")

    ax3.plot([0, 1], [0, 1], color="#FFFFFF", ls="--", lw=1, alpha=0.6)
    ax3.set_title("Calibration Reliability Curve", color=EMERALD, fontweight="bold")
    ax3.set_xlabel("Mean Predicted Prob"); ax3.set_ylabel("Observed Fraction")
    ax3.grid(ls=":", alpha=0.5); ax3.legend(frameon=False, fontsize=7.6, loc="upper left")

    plt.tight_layout(rect=[0, 0.02, 1, 0.91])
    fig.savefig(FIG_DIR / "02_model_benchmark_calibration.png", facecolor=BG)
    plt.close(fig)

    # --- Figure 3: Global SHAP Explainability ---
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    fig.patch.set_facecolor(BG)
    top_f = [clean_names[i] for i in top_idx][::-1]
    top_v = [mean_abs_shap[i] for i in top_idx][::-1]
    # Direction sign via correlation between feature value and SHAP value
    dirs = []
    for i in top_idx[::-1]:
        corr = np.corrcoef(X_test_trans[:, i], shap_vals[:, i])[0, 1]
        dirs.append(ROSE if (not np.isnan(corr) and corr > 0) else EMERALD)

    y_pos = np.arange(len(top_f))
    ax.barh(y_pos, top_v, color=dirs, height=0.62, alpha=0.9)
    for idx_b, val in enumerate(top_v):
        ax.text(val + 0.008, idx_b, f"{val:.3f}", va="center", color=TEXT, fontsize=8.8, family="monospace")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_f, color=TEXT, fontsize=9.2)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Churn Log-Odds)")
    ax.set_title(f"Top 12 Global Churn Drivers ({best_name} · SHAP Explainability)", color=TEXT, fontsize=13.5, fontweight="bold", pad=12)
    ax.text(0.98, 0.06, "■ Red = Increases Churn Risk   ■ Green = Reduces Churn Risk",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.8, family="monospace")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_xlim(0, max(top_v) * 1.18)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_shap_explainability.png", facecolor=BG)
    plt.close(fig)

    # --- Figure 4: Financial Retention ROI Curve + Confusion Matrix ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=160, gridspec_kw={"width_ratios": [1.45, 1]})
    fig.patch.set_facecolor(BG)
    fig.suptitle("Business Impact — Profit-Maximizing Threshold Optimization (Test Set)", color=TEXT, fontsize=14.5, fontweight="bold", y=0.97)

    test_rois = [compute_retention_roi(y_test, test_probs[best_name], meta_test["MonthlyCharges"].values, t) for t in thresholds]
    profits = [r["net_profit_usd"] / 1000.0 for r in test_rois]
    ax1.plot(thresholds, profits, color=EMERALD, lw=2.5, label="Net Retention Profit ($K)")
    ax1.fill_between(thresholds, profits, 0, where=np.array(profits) > 0, color=EMERALD, alpha=0.14)
    ax1.axvline(0.50, color=AMBER, ls="--", lw=1.6, label=f"Default 0.50 (${roi_default_test['net_profit_usd']/1000:.1f}K)")
    ax1.axvline(best_t_val, color=CYAN, ls="-", lw=2.0, label=f"Val-Optimized {best_t_val:.2f} (${roi_opt_test['net_profit_usd']/1000:.1f}K)")
    ax1.axhline(0, color=MUTED, lw=0.8)
    ax1.set_xlabel("Classification Probability Threshold")
    ax1.set_ylabel("Net 12-Month Retention Profit ($ Thousands)")
    ax1.set_title("Expected Profit Curve ($100 Offer · 35% Save Rate)", color=CYAN, fontweight="bold")
    ax1.grid(ls=":", alpha=0.5)
    ax1.legend(frameon=False, fontsize=8.6)

    cm = confusion_matrix(y_test, (test_probs[best_name] >= best_t_val).astype(int))
    ax2.imshow(cm, cmap="Blues", alpha=0.78)
    labels = ["Stay (0)", "Churn (1)"]
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(labels, color=TEXT); ax2.set_yticklabels(labels, color=TEXT)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    ax2.set_title(f"Confusion Matrix @ Threshold {best_t_val:.2f}", color=VIOLET, fontweight="bold")
    for r_i in range(2):
        for c_i in range(2):
            ax2.text(c_i, r_i, f"{cm[r_i, c_i]:,}", ha="center", va="center", color="#FFFFFF", fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.91])
    fig.savefig(FIG_DIR / "04_retention_roi_threshold.png", facecolor=BG)
    plt.close(fig)

    print("[5/5] Pipeline complete! All metrics, SQL DB, action list, and 4 figures saved.")


if __name__ == "__main__":
    main()

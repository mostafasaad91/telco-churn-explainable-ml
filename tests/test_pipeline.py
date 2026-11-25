import json
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_outputs_and_integrity():
    metrics_path = ROOT / "outputs" / "metrics.json"
    assert metrics_path.exists(), "metrics.json missing"
    m = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Zero customers dropped (tenure=0 TotalCharges imputed as 0.0 in SQL)
    assert m["qa_summary"]["total_rows"] == 7043
    assert m["qa_summary"]["unique_customers"] == 7043
    assert m["qa_summary"]["blank_total_charges"] == 11

    # Strict split sum check
    s = m["splits"]
    assert s["train"] + s["val"] + s["test"] == 7043

    # Selected ML model significantly outperforms naive baseline
    best = m["selected_model"]
    assert m["models"][best]["test_roc_auc"] >= 0.82
    assert m["test_roi_at_optimal"]["net_profit_usd"] > m["test_roi_at_50"]["net_profit_usd"]

    # SQLite DB views exist and have zero nulls in TotalCharges
    conn = sqlite3.connect(ROOT / "data" / "telco_churn.db")
    df = pd.read_sql_query("SELECT * FROM vw_churn_features", conn)
    conn.close()
    assert len(df) == 7043
    assert df["TotalCharges"].isna().sum() == 0

    # All 4 AMOLED figures exist
    for fig in [
        "01_churn_eda_drivers.png",
        "02_model_benchmark_calibration.png",
        "03_shap_explainability.png",
        "04_retention_roi_threshold.png",
    ]:
        p = ROOT / "outputs" / "figures" / fig
        assert p.exists() and p.stat().st_size > 20_000, f"Figure missing/small: {fig}"

    print("ALL TELCO CHURN PIPELINE TESTS PASSED!")


if __name__ == "__main__":
    test_outputs_and_integrity()

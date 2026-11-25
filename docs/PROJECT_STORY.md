# Project Story — Why, How & Challenges Faced (EN & AR)

## 1. Why We Built This Project (ليه بنعمل المشروع ده؟)

- **Business Problem:** In subscription businesses (Telecom / SaaS), acquiring a new customer costs 5–7× more than retaining an existing one. In the IBM Telco dataset, **26.54% of customers (1,869 of 7,043)** churned — concentrated heavily in Month-to-Month contracts (**42.6% churn**) and first-6-month customers (**55.6% churn**).
- **Portfolio Value:** While my *AdventureWorks 2022* project demonstrates SQL BI + Unsupervised ML (K-Means RFM), this project demonstrates **Supervised Classification, Probability Calibration, SHAP Explainability, Slice Error Auditing, and Financial Threshold Optimization**.

> **الملخص بالعربية:** هدف المشروع مش بس تدريب نموذج يتنبأ مين هيمشي بدقة عالية، لكن **ربط التوقع بقرار مالي حقيقي**: نوجّه عرض الاحتفاظ ($100) لمين بالظبط عشان نعظّم صافي الربح، ونشرح لفريق الـ CRM سبب خطورة كل عميل على حدة عبر **SHAP**.

---

## 2. How It Was Built Step-by-Step (إزاي إتبنى خطوة بخطوة؟)

1. **SQL Ingestion & Quality Audit (`sql/01_schema_and_views.sql`):**
   - Loaded the raw CSV into SQLite (`raw_telco_churn`) as strings first to catch hidden schema anomalies.
   - Created `vw_churn_qa`, `vw_churn_features`, and `vw_segment_summary`.
2. **Leakage-Free Feature Engineering & Splits (`src/run_pipeline.py`):**
   - Engineered business ratios: `active_services_count`, `charge_per_service`, `price_hike_ratio`, `is_electronic_check`, `has_security_or_support`.
   - Split data into **60% Train (4,225) / 20% Validation (1,409) / 20% Test (1,409)** stratified by churn.
   - Fit all scaling (`StandardScaler`) and encoding (`OneHotEncoder`) strictly inside scikit-learn `Pipeline` objects.
3. **5-Model Benchmark + Bootstrap CIs:**
   - Benchmarked Naive Baseline, Logistic Regression, Random Forest, LightGBM, and XGBoost on identical 5-fold Stratified CV splits.
   - Selected **XGBoost** strictly using **Validation ROC-AUC (`0.8384`)**, then evaluated once on the untouched Test set (**Test ROC-AUC `0.8478`**, 95% CI `[0.8256, 0.8688]`, **Brier `0.1347`**).
4. **SHAP Explainability & Actionable CRM Export:**
   - Computed `TreeExplainer` SHAP values on the Test set to rank global drivers (`contract_months`, `tenure`, `MonthlyCharges`, `Fiber optic`, `OnlineSecurity_No`, `is_electronic_check`) and exported `outputs/high_risk_action_list.csv` with top-3 individual drivers per customer.
5. **Financial ROI Threshold Optimization:**
   - Swept thresholds `[0.08 .. 0.85]` on the **Validation set** to find the profit-maximizing cutoff (`0.37`), then proved its lift on the held-out Test set (**$31,835 → $40,492 net profit, +27.2%**).
6. **Automated Verification (`tests/test_pipeline.py`):**
   - Automated tests verify zero dropped rows (`7,043`), zero null `TotalCharges`, ROC-AUC ≥ 0.82, positive ROI lift, and all 4 generated AMOLED charts.

---

## 3. Key Questions & Challenges We Faced (إيه الأسئلة والتحديات اللي واجهتنا وحليناها إزاي؟)

### Q1: `TotalCharges` had 11 blank string values — should we drop those 11 rows?
- **Investigation:** Running `vw_churn_qa` revealed that **all 11 blank `TotalCharges` rows had `tenure = 0`** (brand-new signups before their first billing invoice!).
- **Decision:** Dropping them would bias the model against brand-new signups (who have the highest early-churn risk!). Instead, we imputed `TotalCharges = 0.0` in SQL AND created an explicit `total_charges_was_missing` / `is_new_customer` indicator feature, retaining **100% of all 7,043 customers**.

### Q2: Why not evaluate and optimize the decision threshold using Accuracy at `0.50`?
- **Problem:** With 26.5% churn imbalance, a naive classifier predicting "Stay" for everyone gets 73.5% accuracy while catching 0 churners. Even a good model at `0.50` misses profitable retention targets because losing a `$80/month` customer (`$960/yr`) hurts far more than spending `$100` on a false positive.
- **Solution:** Modeled exact unit economics (`$100` cost, `35%` save rate, `12 × MonthlyCharges` LTV), optimized the cutoff strictly on Validation (`0.37`), and validated a **+$8,657 (+27.2%) net profit gain** on the unseen Test set.

### Q3: Does the model perform equally well across customer slices? (Slice Error Audit)
- **Finding:** On the held-out Test set:
  - **Month-to-month (`42.6%` churn):** High recall (`80.9%`) and precision (`59.5%`) — captures the vast majority of revenue leakage.
  - **0–6 month tenure (`55.6%` churn):** `85.5%` recall and `69.2%` precision.
  - **Two-year contracts (`2.7%` churn):** Strong ranking AUC (`0.814`), but `0.0%` recall at `0.37` because predicted probabilities naturally stay below `0.37` (meaning expensive `$100` offers aren't wasted on locked-in low-risk accounts!).

### Q4: Why do Fiber Optic customers churn much faster than DSL?
- **SHAP Evidence:** `InternetService_Fiber optic` (`0.208` mean |SHAP|) and `MonthlyCharges` (`0.272`) both push churn UP, while `OnlineSecurity_No` (`0.202`) and `TechSupport_No` (`0.113`) amplify risk.
- **Business Recommendation:** Bundle free 3-month **Online Security + Tech Support** for new Month-to-Month Fiber Optic customers during months 1–6 and incentivize autopay migration away from Electronic Check (`+0.128` SHAP risk).

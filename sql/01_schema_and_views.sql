-- ============================================================================
-- IBM Telco Customer Churn — SQLite Analytical & Feature Engineering Views
-- Author: Mostafa Saad
-- ============================================================================

DROP VIEW IF EXISTS vw_churn_qa;
CREATE VIEW vw_churn_qa AS
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customerID) AS unique_customers,
    SUM(CASE WHEN TRIM(COALESCE(TotalCharges, '')) = '' THEN 1 ELSE 0 END) AS blank_total_charges,
    SUM(CASE WHEN tenure = 0 THEN 1 ELSE 0 END) AS zero_tenure_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM raw_telco_churn;

DROP VIEW IF EXISTS vw_churn_features;
CREATE VIEW vw_churn_features AS
SELECT
    customerID,
    gender,
    CAST(SeniorCitizen AS INTEGER) AS SeniorCitizen,
    Partner,
    Dependents,
    CAST(tenure AS INTEGER) AS tenure,
    CASE WHEN CAST(tenure AS INTEGER) = 0 THEN 1 ELSE 0 END AS is_new_customer,
    CASE
        WHEN CAST(tenure AS INTEGER) <= 6  THEN '0-6m'
        WHEN CAST(tenure AS INTEGER) <= 12 THEN '7-12m'
        WHEN CAST(tenure AS INTEGER) <= 24 THEN '13-24m'
        WHEN CAST(tenure AS INTEGER) <= 48 THEN '25-48m'
        ELSE '49m+'
    END AS tenure_cohort,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    CASE Contract
        WHEN 'Month-to-month' THEN 1
        WHEN 'One year' THEN 12
        WHEN 'Two year' THEN 24
        ELSE 1
    END AS contract_months,
    PaperlessBilling,
    PaymentMethod,
    CASE WHEN PaymentMethod = 'Electronic check' THEN 1 ELSE 0 END AS is_electronic_check,
    CASE WHEN PaymentMethod LIKE '%automatic%' THEN 1 ELSE 0 END AS is_autopay,
    CAST(MonthlyCharges AS REAL) AS MonthlyCharges,
    -- Preserve tenure=0 rows instead of dropping: TotalCharges is 0.0 before first billing cycle
    CASE
        WHEN TRIM(COALESCE(TotalCharges, '')) = '' THEN 0.0
        ELSE CAST(TotalCharges AS REAL)
    END AS TotalCharges,
    CASE WHEN TRIM(COALESCE(TotalCharges, '')) = '' THEN 1 ELSE 0 END AS total_charges_was_missing,
    (
        (CASE WHEN PhoneService = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN MultipleLines = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END)
    ) AS active_services_count,
    ROUND(CAST(MonthlyCharges AS REAL) * 12.0, 2) AS annual_revenue_12m,
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS churn_flag
FROM raw_telco_churn;

DROP VIEW IF EXISTS vw_segment_summary;
CREATE VIEW vw_segment_summary AS
SELECT
    Contract,
    InternetService,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN annual_revenue_12m ELSE 0 END), 2) AS annual_rev_lost
FROM vw_churn_features
GROUP BY Contract, InternetService
ORDER BY churn_rate_pct DESC;

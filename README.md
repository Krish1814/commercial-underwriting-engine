# Production Runbook: Commercial Underwriting Engine & Pre-Disbursal ATO Fraud Gateway
Internal System Documentation | Architecture Identifier: COMM_UNDERWRITING_PROD_V4

## Core Operational Objective
Real-time embedded lending underwriting pipeline designed to evaluate credit eligibility and mitigate Account Takeover (ATO) risk at the point of instant capital disbursal. The system operates under a strict P99 latency SLA of <2,000ms.

## System Topology & Event Flow Sequence
1. Compliance Gatekeeper (Level 1)
   - Checks global account status indicators.
   - Evaluates geographical launch parameters against whitelist arrays.
   - Drops invalid records early to preserve downstream computational capacity.

2. Feature Engineering & Multi-Model Ingestion (Level 2 & 3)
   - Ingests pre-calculated output vectors from downstream platform models (Risk/Churn scores).
   - Coordinates multi-family continuous and categorical metrics using custom ColumnTransformer logic.
   - Implements median numerical imputation to insulate against heavy-tailed financial distribution noise.

3. Loss-Optimized Underwriting Selection (Level 4 & 5)
   - Computes Probability of Default (PD) floats via a cost-sensitive classification ensemble.
   - Leverages calculated scale_pos_weight multipliers to offset severe class imbalance.
   - Triage routing based on hard risk boundaries: Auto-Approve (PD <= 0.03) vs. Auto-Decline (PD >= 0.12).

4. Pre-Deposit Post-Underwriting Fraud Gateway (Level 6)
   - Evaluates security modification event timestamps right before instant transfer execution.
   - Triggers deterministic hard blocks on recent infrastructure changes (<=72 hours).
   - Coordinates 15-minute step-up SMS OTP authentication sequences for medium-velocity anomalies.

## Production Schema Data Taxonomy
The framework maintains a 30-feature schema partitioned into localized risk vectors:
* Financial Risk Vectors: Metrics tracking cash-flow volatility, liquidity depth, and debt servicing coverage (e.g., monthly_revenue, revenue_coefficient_of_variation, debt_service_coverage_ratio).
* Credit History Vectors: Index parameters detailing bureau file maturity, utilization trends, and historical repayment issues (e.g., fico_score, revolving_trended_utilization_delta, delinquencies_12m).
* Behavioral/Operational Vectors: Real-time payment processor telemetry and client-side device signals (e.g., historical_chargeback_rate, failed_payment_rate_30d, missing_device_telemetry).

## Post-Deployment Observability Framework
The deployment pipeline logs a standardized JSON record for every transaction to active telemetry tables. Performance health is tracked via rolling 30-day analytics windows:

* Customer Insult Rate (False Positive Rate): Monitored to evaluate legitimate user disruption. Target constraint: <1.50%.
* Bad-Debt Loss Leakage Rate (False Negative Rate): Monitored to trace missed default events. Target constraint: <2.00%.
* Population Stability Index (PSI): Tracked to detect systemic data drift. Values >=0.25 trigger an automated model retraining flag.

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

class EnterpriseUnderwritingEngine:
    def __init__(self, config_path=None):
        self.target_col = 'default_flag'
        self.preprocessor = None
        self.feature_names = None
        self.xgb_model = None
        self.rf_model = None
        
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "eligible_states": ['CA', 'NY', 'TX', 'IL', 'FL'],
                "hard_lock_codes": ['FRD', 'FRZ', 'FRAUD']
            }

        self.financial_features = [
            'monthly_revenue', 'revenue_growth_6m', 'revenue_volatility_6m',
            'cash_balance', 'total_debt', 'monthly_debt_payment',
            'debt_service_coverage_ratio', 'current_ratio', 'business_age_months'
        ]
        self.credit_history_features = [
            'fico_score', 'credit_history_months', 'num_open_credit_accounts',
            'num_delinquent_accounts', 'delinquencies_12m', 'days_past_due_max_12m',
            'previous_defaults', 'utilization_rate', 'credit_limit'
        ]
        self.behavioral_features = [
            'transaction_velocity_30d', 'avg_transaction_amount_30d', 
            'transaction_volume_30d', 'transaction_volume_change_30d',
            'historical_chargeback_rate', 'refund_rate_30d', 'failed_payment_rate_30d',
            'bank_balance_volatility_90d', 'negative_balance_days_90d', 
            'missing_device_telemetry'
        ]
        self.categorical_features = ['industry_code']

    def check_compliance_limits(self, account_status, reason_code, state):
        if account_status == 'LOCKED' and reason_code in self.config["hard_lock_codes"]:
            return False, "HARD_EXCLUSION_FRAUD_LOCK"
        if state not in self.config["eligible_states"]:
            return False, "REGIONAL_EXCLUSION_OUT_OF_BOUNDS"
        return True, "PROCEED_TO_UNDERWRITING"

    def generate_base_data_matrix(self, samples=50000, seed=42):
        np.random.seed(seed)
        merchant_ids = np.arange(500000, 500000 + samples)
        business_age_months = np.random.randint(6, 240, size=samples)
        industry_code = np.random.choice(['Retail', 'SaaS', 'Restaurant', 'Construction', 'Healthcare'], size=samples, p=[0.3, 0.2, 0.2, 0.15, 0.15])
        
        monthly_revenue = np.random.exponential(scale=60000, size=samples) + 5000
        revenue_growth_6m = np.random.normal(loc=0.01, scale=0.12, size=samples)
        revenue_volatility_6m = np.random.beta(a=2, b=5, size=samples)
        total_debt = monthly_revenue * np.random.uniform(0.5, 4.0, size=samples)
        monthly_debt_payment = (total_debt * np.random.uniform(0.015, 0.025, size=samples)).round(2)
        cash_balance = (monthly_revenue * np.random.uniform(0.2, 1.5, size=samples) * (1 + revenue_growth_6m)).clip(lower=0)
        dscr = (monthly_revenue * 0.3) / np.clip(monthly_debt_payment, 1, None)
        current_ratio = np.random.lognormal(mean=0.3, sigma=0.4, size=samples)
        
        credit_history_months = (business_age_months * np.random.uniform(0.6, 1.1, size=samples)).astype(int).clip(6, 300)
        utilization_rate = np.random.uniform(0.05, 0.95, size=samples)
        credit_limit = (monthly_revenue * np.random.uniform(1.0, 3.0, size=samples)).round(-3)
        num_delinquent_accounts = np.random.poisson(lam=0.18, size=samples)
        delinquencies_12m = np.where(num_delinquent_accounts > 0, np.random.poisson(lam=1.0, size=samples) + 1, 0)
        days_past_due_max_12m = np.where(delinquencies_12m > 0, np.random.choice([0, 30, 60, 90], size=samples, p=[0.5, 0.3, 0.15, 0.05]), 0)
        previous_defaults = np.random.choice([0, 1], size=samples, p=[0.97, 0.03])
        
        fico_score = (np.random.normal(loc=715, scale=35, size=samples) - (num_delinquent_accounts * 40) - (previous_defaults * 130)).astype(int).clip(300, 850)
        num_open_credit_accounts = np.random.poisson(lam=5, size=samples) + 1
        
        transaction_velocity_30d = np.random.poisson(lam=280, size=samples)
        avg_transaction_amount_30d = (monthly_revenue / np.clip(transaction_velocity_30d, 1, None)).round(2)
        transaction_volume_30d = (transaction_velocity_30d * avg_transaction_amount_30d)
        transaction_volume_change_30d = revenue_growth_6m + np.random.normal(0, 0.04, size=samples)
        historical_chargeback_rate = np.random.beta(a=1, b=85, size=samples)
        refund_rate_30d = np.random.beta(a=2, b=45, size=samples)
        failed_payment_rate_30d = np.random.beta(a=2, b=55, size=samples)
        bank_balance_volatility_90d = revenue_volatility_6m * np.random.uniform(0.8, 1.3, size=samples)
        negative_balance_days_90d = np.where(cash_balance < (monthly_revenue * 0.08), np.random.poisson(lam=5, size=samples), 0)
        missing_device_telemetry = np.random.choice([0, 1], size=samples, p=[0.94, 0.06])
        
        industry_penalty = np.where(industry_code == 'Restaurant', 0.90, np.where(industry_code == 'Construction', 0.60, 0.0))
        logit_link = (-0.016 * (fico_score - 710) + 3.6 * utilization_rate + 4.5 * revenue_volatility_6m - 2.8 * revenue_growth_6m + 0.09 * days_past_due_max_12m + 5.8 * historical_chargeback_rate + 6.5 * failed_payment_rate_30d + 0.50 * negative_balance_days_90d + 1.6 * previous_defaults + 1.2 * missing_device_telemetry + industry_penalty - 5.8)
        default_flag = np.random.binomial(1, 1.0 / (1.0 + np.exp(-logit_link)))
        
        return pd.DataFrame({
            'merchant_id': merchant_ids, 'business_age_months': business_age_months, 'industry_code': industry_code,
            'monthly_revenue': monthly_revenue, 'revenue_growth_6m': revenue_growth_6m, 'revenue_volatility_6m': revenue_volatility_6m,
            'cash_balance': cash_balance, 'total_debt': total_debt, 'monthly_debt_payment': monthly_debt_payment,
            'debt_service_coverage_ratio': dscr, 'current_ratio': current_ratio, 'fico_score': fico_score,
            'credit_history_months': credit_history_months, 'num_open_credit_accounts': num_open_credit_accounts,
            'num_delinquent_accounts': num_delinquent_accounts, 'delinquencies_12m': delinquencies_12m,
            'days_past_due_max_12m': days_past_due_max_12m, 'previous_defaults': previous_defaults,
            'utilization_rate': utilization_rate, 'credit_limit': credit_limit, 'transaction_velocity_30d': transaction_velocity_30d,
            'avg_transaction_amount_30d': avg_transaction_amount_30d, 'transaction_volume_30d': transaction_volume_30d,
            'transaction_volume_change_30d': transaction_volume_change_30d, 'historical_chargeback_rate': historical_chargeback_rate,
            'refund_rate_30d': refund_rate_30d, 'failed_payment_rate_30d': failed_payment_rate_30d,
            'bank_balance_volatility_90d': bank_balance_volatility_90d, 'negative_balance_days_90d': negative_balance_days_90d,
            'missing_device_telemetry': missing_device_telemetry, 'default_flag': default_flag
        })

    def execute_transform_pipeline(self, X_train):
        numeric_cols = self.financial_features + self.credit_history_features + self.behavioral_features
        
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num_transform', num_transformer, numeric_cols),
            ('cat_transform', cat_transformer, self.categorical_features)
        ])
        self.preprocessor.fit(X_train)
        
        cat_cols = self.preprocessor.named_transformers_['cat_transform'].named_steps['onehot'].get_feature_names_out(self.categorical_features).tolist()
        self.feature_names = numeric_cols + cat_cols

    def fit_estimators(self, X_train, y_train):
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_ratio = neg_count / max(pos_count, 1)
        
        self.xgb_model = XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5,
            scale_pos_weight=scale_ratio, random_state=42, eval_metric='logloss'
        )
        self.xgb_model.fit(X_train, y_train)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100, class_weight='balanced', max_depth=10,
            random_state=42, n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)

    def evaluate_pipeline_performance(self, X_test, y_test):
        probs = self.xgb_model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)
        return roc_auc_score(y_test, probs), classification_report(y_test, preds, output_dict=True)

    def eval_pre_disbursal_fraud_gate(self, metrics, challenge_passed=True):
        if metrics.get('hours_since_renumber', 999) <= 72: return "DECLINE_ATO_RENUMBER_72H"
        if metrics.get('hours_since_shared_device', 999) <= 72: return "DECLINE_ATO_SHARED_DEVICE_72H"
        if metrics.get('hours_since_new_device', 999) <= 72: return "DECLINE_ATO_NEW_DEVICE_72H"
        if metrics.get('hours_since_unlock', 999) <= 72: return "DECLINE_ATO_ACCOUNT_UNLOCK_72H"
        if metrics.get('hours_since_phone_change', 999) <= 24 or metrics.get('hours_since_recovery', 999) <= 24:

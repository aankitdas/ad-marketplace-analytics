"""
Churn Prediction Model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb


class ChurnPredictor:
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
    
    def prepare_features(self, advertisers: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for churn prediction.
        """
        # Aggregate event-level metrics per advertiser
        advertiser_metrics = events.groupby('advertiser_id').agg({
            'price_paid': ['sum', 'mean', 'std'],
            'clicked': ['sum', 'mean'],
            'converted': ['sum', 'mean'],
            'impression_id': 'count',
            'day': ['min', 'max', 'nunique']
        }).reset_index()
        
        # Flatten column names
        advertiser_metrics.columns = [
            'advertiser_id', 
            'total_spend', 'avg_price', 'std_price',
            'total_clicks', 'ctr',
            'total_conversions', 'cvr',
            'total_impressions',
            'first_day', 'last_day', 'active_days'
        ]
        
        # Merge with advertiser data
        df = advertisers.merge(advertiser_metrics, on='advertiser_id', how='left')
        
        # Fill NaN for advertisers with no events
        df = df.fillna(0)
        
        # Create additional features
        df['days_since_signup'] = df['last_day'] - df['signup_day']
        df['spend_per_day'] = df['total_spend'] / (df['active_days'] + 1)
        df['budget_utilization'] = df['spend_per_day'] / df['daily_budget']
        df['impressions_per_day'] = df['total_impressions'] / (df['active_days'] + 1)
        
        # Encode categorical variables
        for col in ['vertical', 'bid_strategy']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def train(self, advertisers: pd.DataFrame, events: pd.DataFrame) -> dict:
        """
        Train churn prediction model.
        """
        # Prepare features
        df = self.prepare_features(advertisers, events)
        
        # Define features
        self.feature_names = [
            'daily_budget', 'quality_score', 'signup_day',
            'total_spend', 'avg_price', 'total_clicks', 'ctr',
            'total_conversions', 'total_impressions', 'active_days',
            'spend_per_day', 'budget_utilization', 'impressions_per_day',
            'vertical_encoded', 'bid_strategy_encoded'
        ]
        
        X = df[self.feature_names]
        y = df['is_churned'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='auc'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def full_report(self, advertisers: pd.DataFrame, events: pd.DataFrame) -> None:
        """Print full model report."""
        
        print("=" * 60)
        print("CHURN PREDICTION MODEL REPORT")
        print("=" * 60)
        
        print("\n1. TRAINING MODEL...")
        print("-" * 40)
        metrics = self.train(advertisers, events)
        
        print(f"\n2. MODEL PERFORMANCE")
        print("-" * 40)
        print(f"AUC-ROC:   {metrics['auc_roc']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        
        print(f"\n3. CONFUSION MATRIX")
        print("-" * 40)
        cm = metrics['confusion_matrix']
        print(f"                 Predicted")
        print(f"              No Churn  Churn")
        print(f"Actual No Churn   {cm[0][0]:4d}   {cm[0][1]:4d}")
        print(f"Actual Churn      {cm[1][0]:4d}   {cm[1][1]:4d}")
        
        print(f"\n4. FEATURE IMPORTANCE (Top 10)")
        print("-" * 40)
        importance = self.feature_importance()
        for _, row in importance.head(10).iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"{row['feature']:25s} {row['importance']:.3f} {bar}")
        
        print("\n" + "=" * 60)


# Test it
if __name__ == '__main__':
    # Load saved data
    events = pd.read_parquet('data/events.parquet')
    advertisers = pd.read_parquet('data/advertisers.parquet')
    
    predictor = ChurnPredictor()
    predictor.full_report(advertisers, events)
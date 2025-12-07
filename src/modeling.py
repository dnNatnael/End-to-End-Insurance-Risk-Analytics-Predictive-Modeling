"""
Machine learning modeling module for insurance risk prediction.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for insurance data."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for modeling.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Car age (from RegistrationYear)
        if 'RegistrationYear' in df.columns:
            current_year = pd.to_datetime(df['TransactionMonth']).dt.year.max() if 'TransactionMonth' in df.columns else 2015
            df['CarAge'] = current_year - pd.to_numeric(df['RegistrationYear'], errors='coerce')
            df['CarAge'] = df['CarAge'].clip(lower=0, upper=50)  # Cap at 50 years
        
        # Model age (from VehicleIntroDate)
        if 'VehicleIntroDate' in df.columns:
            try:
                df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], format='%m/%Y', errors='coerce')
                intro_year = df['VehicleIntroDate'].dt.year
                reg_year = pd.to_numeric(df['RegistrationYear'], errors='coerce')
                df['ModelAge'] = reg_year - intro_year
                df['ModelAge'] = df['ModelAge'].clip(lower=0, upper=30)
            except:
                df['ModelAge'] = np.nan
        
        # Risk ratio features
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            df['LossRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1e-6)
            df['ClaimsToSumInsured'] = df['TotalClaims'] / (df['SumInsured'] + 1e-6)
        
        # Premium per unit sum insured
        if 'SumInsured' in df.columns and 'TotalPremium' in df.columns:
            df['PremiumPerSumInsured'] = df['TotalPremium'] / (df['SumInsured'] + 1e-6)
        
        # High-risk indicators
        if 'Province' in df.columns:
            # Calculate province-level loss ratios
            province_loss_ratios = df.groupby('Province').apply(
                lambda x: x['TotalClaims'].sum() / (x['TotalPremium'].sum() + 1e-6)
            )
            high_risk_threshold = province_loss_ratios.quantile(0.75)
            df['HighRiskProvince'] = df['Province'].map(
                lambda x: 1 if province_loss_ratios.get(x, 0) > high_risk_threshold else 0
            )
        
        if 'PostalCode' in df.columns:
            # High-risk zip codes (top 10% by loss ratio)
            zip_loss_ratios = df.groupby('PostalCode').apply(
                lambda x: x['TotalClaims'].sum() / (x['TotalPremium'].sum() + 1e-6)
            )
            high_risk_zip_threshold = zip_loss_ratios.quantile(0.90)
            df['HighRiskZip'] = df['PostalCode'].map(
                lambda x: 1 if zip_loss_ratios.get(x, 0) > high_risk_zip_threshold else 0
            )
        
        # Vehicle risk indicators
        if 'VehicleType' in df.columns:
            vehicle_loss_ratios = df.groupby('VehicleType').apply(
                lambda x: x['TotalClaims'].sum() / (x['TotalPremium'].sum() + 1e-6)
            )
            high_risk_vehicle_threshold = vehicle_loss_ratios.quantile(0.75)
            df['HighRiskVehicleType'] = df['VehicleType'].map(
                lambda x: 1 if vehicle_loss_ratios.get(x, 0) > high_risk_vehicle_threshold else 0
            )
        
        # Binary flags
        if 'AlarmImmobiliser' in df.columns:
            df['HasAlarm'] = (df['AlarmImmobiliser'] == 'Yes').astype(int)
        if 'TrackingDevice' in df.columns:
            df['HasTracking'] = (df['TrackingDevice'] == 'Yes').astype(int)
        if 'NewVehicle' in df.columns:
            df['IsNewVehicle'] = (df['NewVehicle'] == 'More than 6 months').astype(int)
        
        # Engine features
        if 'Cylinders' in df.columns:
            df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce')
        if 'cubiccapacity' in df.columns:
            df['cubiccapacity'] = pd.to_numeric(df['cubiccapacity'], errors='coerce')
        if 'kilowatts' in df.columns:
            df['kilowatts'] = pd.to_numeric(df['kilowatts'], errors='coerce')
        
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        drop_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for modeling with encoding and scaling.
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
            categorical_cols: List of categorical columns to encode
            numeric_cols: List of numeric columns to include
            drop_cols: List of columns to drop
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        df = df.copy()
        
        # Drop specified columns
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # Extract target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        y = df[target_col].copy()
        df = df.drop(columns=[target_col])
        
        # Identify columns if not provided
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        id_cols = ['UnderwrittenCoverID', 'PolicyID', 'TransactionMonth']
        categorical_cols = [c for c in categorical_cols if c not in id_cols]
        numeric_cols = [c for c in numeric_cols if c not in id_cols]
        
        # Encode categorical variables (target encoding for high cardinality, one-hot for low)
        encoded_dfs = []
        feature_names = numeric_cols.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            n_unique = df[col].nunique()
            
            if n_unique > 20:  # High cardinality - use target encoding
                if not self.is_fitted:
                    # Calculate target means for encoding
                    if col in self.label_encoders:
                        encoder = self.label_encoders[col]
                    else:
                        encoder = {}
                        for cat in df[col].unique():
                            mask = df[col] == cat
                            if mask.sum() > 0:
                                encoder[cat] = y[mask].mean()
                        self.label_encoders[col] = encoder
                    
                    df[f'{col}_encoded'] = df[col].map(encoder).fillna(y.mean())
                else:
                    encoder = self.label_encoders.get(col, {})
                    df[f'{col}_encoded'] = df[col].map(encoder).fillna(y.mean() if hasattr(y, 'mean') else 0)
                
                feature_names.append(f'{col}_encoded')
            else:  # Low cardinality - use one-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                encoded_dfs.append(dummies)
                feature_names.extend(dummies.columns.tolist())
        
        # Combine all features
        X = df[numeric_cols].copy()
        for dummy_df in encoded_dfs:
            X = pd.concat([X, dummy_df], axis=1)
        
        # Handle missing values
        X = X.fillna(X.median() if len(numeric_cols) > 0 else 0)
        
        # Scale features
        if not self.is_fitted:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            self.is_fitted = True
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled, y, feature_names


class ModelTrainer:
    """Model training and evaluation class."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def train_regression_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate regression models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of model results
        """
        results = {}
        
        # Linear Regression
        logger.info("Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        results['Linear Regression'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'model': lr
        }
        self.models['Linear Regression'] = lr
        
        # Decision Tree
        logger.info("Training Decision Tree Regressor...")
        dt = DecisionTreeRegressor(random_state=self.random_state, max_depth=10)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        results['Decision Tree'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'model': dt
        }
        self.models['Decision Tree'] = dt
        
        # Random Forest
        logger.info("Training Random Forest Regressor...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        results['Random Forest'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'model': rf
        }
        self.models['Random Forest'] = rf
        
        # XGBoost
        logger.info("Training XGBoost Regressor...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        results['XGBoost'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'model': xgb_model
        }
        self.models['XGBoost'] = xgb_model
        
        self.results['regression'] = results
        return results
    
    def train_classification_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate classification models.
        
        Args:
            X_train: Training features
            y_train: Training target (binary)
            X_test: Test features
            y_test: Test target (binary)
            
        Returns:
            Dictionary of model results
        """
        results = {}
        
        # Logistic Regression
        logger.info("Training Logistic Regression...")
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=-1)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        results['Logistic Regression'] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, y_pred_proba),
            'model': lr
        }
        self.models['Logistic Regression'] = lr
        
        # Random Forest Classifier
        logger.info("Training Random Forest Classifier...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        results['Random Forest'] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, y_pred_proba),
            'model': rf
        }
        self.models['Random Forest'] = rf
        
        # XGBoost Classifier
        logger.info("Training XGBoost Classifier...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        results['XGBoost'] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, y_pred_proba),
            'model': xgb_model
        }
        self.models['XGBoost'] = xgb_model
        
        self.results['classification'] = results
        return results
    
    def get_best_model(self, task_type: str, metric: str = None) -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Args:
            task_type: 'regression' or 'classification'
            metric: Metric to optimize (default: R2 for regression, AUC for classification)
            
        Returns:
            Tuple of (model_name, model_object)
        """
        if task_type not in self.results:
            raise ValueError(f"No results found for task type: {task_type}")
        
        results = self.results[task_type]
        
        if task_type == 'regression':
            if metric is None:
                metric = 'R2'
            best_model = max(results.items(), key=lambda x: x[1][metric])
        else:  # classification
            if metric is None:
                metric = 'AUC'
            best_model = max(results.items(), key=lambda x: x[1][metric])
        
        return best_model[0], best_model[1]['model']


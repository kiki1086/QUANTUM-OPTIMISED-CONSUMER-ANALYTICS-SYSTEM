import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import shap

class ModelTrainer:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        
    def train_purchase_prediction(self, X_train, y_train):
        """Trains XGBoost for customer purchase prediction"""
        print("Training XGBoost for Purchase Prediction...")
        self.xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.xgb_model.fit(X_train, y_train)
        return self.xgb_model
        
    def train_demand_forecasting(self, X_train, y_train):
        """Trains LightGBM for market demand forecasting"""
        print("Training LightGBM for Demand Forecasting...")
        self.lgb_model = lgb.LGBMRegressor()
        self.lgb_model.fit(X_train, y_train)
        return self.lgb_model
        
    def generate_explanations(self, model, X):
        """Generates SHAP values for Explainable AI Layer"""
        print("Generating SHAP explanations...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return shap_values

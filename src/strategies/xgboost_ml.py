import pandas as pd
import numpy as np
from typing import Dict, Any
import xgboost as xgb
from src.strategies.base import BaseStrategy

class XGBoostStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="XGBoost_ML", params=params)
    def get_default_params(self): return {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "lookback": 50, "retrain_every": 100}
    def get_param_space(self): return {"n_estimators": ("int", 50, 200, 10), "max_depth": ("int", 2, 6, 1), "learning_rate": ("float", 0.01, 0.3, 0.01), "lookback": ("int", 30, 200, 10), "retrain_every": ("int", 50, 300, 25)}
    
    @staticmethod
    def create_features(data, lookback):
        df = data.copy()
        for lag in [1,3,5,10,20,50]:
            if len(df)>lag: df[f'return_{lag}'] = df['close'].pct_change(lag)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['volume_ratio'] = df['volume']/df['volume'].rolling(20).mean()
        for p in [10,20,50]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
            df[f'dist_sma_{p}'] = (df['close']-df[f'sma_{p}'])/df[f'sma_{p}']
        delta = df['close'].diff()
        gain = delta.where(delta>0,0.0).rolling(14).mean()
        loss = (-delta.where(delta<0,0.0)).rolling(14).mean()
        df['rsi'] = 100-(100/(1+gain/loss))
        df['target'] = (df['close'].shift(-1)>df['close']).astype(int)
        return df.dropna()
    
    def generate_signals(self, data):
        lookback, retrain = self.params["lookback"], self.params["retrain_every"]
        if len(data) < lookback+retrain: return pd.Series(0, index=data.index)
        df = self.create_features(data, lookback)
        feature_cols = [c for c in df.columns if c not in ['open','high','low','close','volume','target']]
        signals = pd.Series(0, index=data.index)
        model, last_retrain = None, lookback
        for i in range(lookback, len(df)-1):
            if i-last_retrain >= retrain or i == lookback:
                train_data = df.iloc[last_retrain:i]
                if len(train_data) > 50:
                    X_train, y_train = train_data[feature_cols], train_data['target']
                    model = xgb.XGBClassifier(n_estimators=self.params["n_estimators"], max_depth=self.params["max_depth"], learning_rate=self.params["learning_rate"], verbosity=0)
                    model.fit(X_train, y_train)
                    last_retrain = i
            if model is not None:
                X_pred = df.iloc[[i]][feature_cols]
                try:
                    proba = model.predict_proba(X_pred)[0]
                    pred = 1 if proba[1]>0.55 else (-1 if proba[0]>0.55 else 0)
                    idx = df.index[i]
                    if idx in signals.index: signals[idx] = pred
                except: pass
        return signals
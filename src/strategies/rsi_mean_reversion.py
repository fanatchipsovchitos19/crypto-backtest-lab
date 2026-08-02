import pandas as pd
import numpy as np
from typing import Dict, Any
from src.strategies.base import BaseStrategy

class RSIMeanReversionStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="RSI_MeanReversion", params=params)
    def get_default_params(self): return {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70}
    def get_param_space(self): return {"rsi_period": ("int", 7, 28, 1), "oversold_threshold": ("int", 20, 40, 5), "overbought_threshold": ("int", 60, 80, 5)}
    
    @staticmethod
    def calculate_rsi(data, period):
        delta = data.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        for i in range(period, len(avg_gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1]*(period-1) + gain.iloc[i])/period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1]*(period-1) + loss.iloc[i])/period
        rs = avg_gain / avg_loss
        return 100 - (100/(1+rs))
    
    def generate_signals(self, data):
        period = self.params["rsi_period"]
        if len(data) < period+1: return pd.Series(0, index=data.index)
        df = data[['close']].copy()
        df['rsi'] = self.calculate_rsi(df['close'], period)
        df['signal'] = 0
        df.loc[(df['rsi'] > self.params["oversold_threshold"]) & (df['rsi'].shift(1) <= self.params["oversold_threshold"]) & df['rsi'].notna(), 'signal'] = 1
        df.loc[(df['rsi'] < self.params["overbought_threshold"]) & (df['rsi'].shift(1) >= self.params["overbought_threshold"]) & df['rsi'].notna(), 'signal'] = -1
        return df['signal']
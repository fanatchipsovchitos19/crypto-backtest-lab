import pandas as pd
from typing import Dict, Any
from src.strategies.base import BaseStrategy

class MACDStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="MACD_Classic", params=params)
    def get_default_params(self): return {"fast_period": 12, "slow_period": 26, "signal_period": 9}
    def get_param_space(self): return {"fast_period": ("int", 8, 20, 1), "slow_period": ("int", 21, 40, 1), "signal_period": ("int", 5, 15, 1)}
    
    def generate_signals(self, data):
        if len(data) < self.params["slow_period"] + self.params["signal_period"]: return pd.Series(0, index=data.index)
        df = data[['close']].copy()
        ema_fast = df['close'].ewm(span=self.params["fast_period"], adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.params["slow_period"], adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['signal_line'] = df['macd'].ewm(span=self.params["signal_period"], adjust=False).mean()
        df['signal'] = 0
        df.loc[(df['macd'] > df['signal_line']) & (df['macd'].shift(1) <= df['signal_line'].shift(1)) & df['signal_line'].notna(), 'signal'] = 1
        df.loc[(df['macd'] < df['signal_line']) & (df['macd'].shift(1) >= df['signal_line'].shift(1)) & df['signal_line'].notna(), 'signal'] = -1
        return df['signal']
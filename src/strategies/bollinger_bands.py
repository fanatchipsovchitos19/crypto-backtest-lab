import pandas as pd
from typing import Dict, Any
from src.strategies.base import BaseStrategy

class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="BollingerBands", params=params)
    def get_default_params(self): return {"bb_period": 20, "bb_std": 2.0}
    def get_param_space(self): return {"bb_period": ("int", 10, 50, 1), "bb_std": ("float", 1.0, 3.0, 0.1)}
    
    def generate_signals(self, data):
        period = self.params["bb_period"]
        if len(data) < period: return pd.Series(0, index=data.index)
        df = data[['close']].copy()
        df['sma'] = df['close'].rolling(period).mean()
        df['std'] = df['close'].rolling(period).std()
        df['upper'] = df['sma'] + self.params["bb_std"] * df['std']
        df['lower'] = df['sma'] - self.params["bb_std"] * df['std']
        df['signal'] = 0
        df.loc[(df['close'] > df['lower']) & (df['close'].shift(1) <= df['lower'].shift(1)) & df['lower'].notna(), 'signal'] = 1
        df.loc[(df['close'] < df['upper']) & (df['close'].shift(1) >= df['upper'].shift(1)) & df['upper'].notna(), 'signal'] = -1
        return df['signal']
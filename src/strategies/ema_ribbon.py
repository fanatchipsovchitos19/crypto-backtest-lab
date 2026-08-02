import pandas as pd
from typing import Dict, Any
from src.strategies.base import BaseStrategy

class EMARibbonStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="EMA_Ribbon", params=params)
    def get_default_params(self): return {"ema1": 8, "ema2": 21, "ema3": 55, "ema4": 89}
    def get_param_space(self): return {"ema1": ("int", 5, 20, 1), "ema2": ("int", 15, 50, 1), "ema3": ("int", 40, 100, 1), "ema4": ("int", 70, 200, 1)}
    
    def generate_signals(self, data):
        if len(data) < self.params["ema4"]: return pd.Series(0, index=data.index)
        df = data[['close']].copy()
        for k in ['ema1','ema2','ema3','ema4']: df[k] = df['close'].ewm(span=self.params[k], adjust=False).mean()
        df['ribbon_up'] = (df['ema1'] > df['ema2']) & (df['ema2'] > df['ema3']) & (df['ema3'] > df['ema4']) & df['ema4'].notna()
        df['signal'] = 0
        df.loc[df['ribbon_up'] & ~df['ribbon_up'].shift(1).fillna(False), 'signal'] = 1
        df.loc[~df['ribbon_up'] & df['ribbon_up'].shift(1).fillna(False), 'signal'] = -1
        return df['signal']
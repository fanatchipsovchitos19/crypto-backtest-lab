import pandas as pd
from typing import Dict, Any
from src.strategies.base import BaseStrategy

class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="SMA_Crossover", params=params)
    def get_default_params(self): return {"fast_period": 20, "slow_period": 50}
    def get_param_space(self): return {"fast_period": ("int", 5, 100, 1), "slow_period": ("int", 20, 300, 1)}
    
    def generate_signals(self, data):
        if len(data) < self.params["slow_period"]: return pd.Series(0, index=data.index)
        df = data[['close']].copy()
        df['sma_fast'] = df['close'].rolling(self.params["fast_period"]).mean()
        df['sma_slow'] = df['close'].rolling(self.params["slow_period"]).mean()
        df['signal'] = 0
        buy = (df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1)) & df['sma_fast'].notna() & df['sma_slow'].notna()
        sell = (df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1)) & df['sma_fast'].notna() & df['sma_slow'].notna()
        df.loc[buy, 'signal'] = 1
        df.loc[sell, 'signal'] = -1
        return df['signal']
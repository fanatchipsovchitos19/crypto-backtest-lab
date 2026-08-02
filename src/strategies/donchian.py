import pandas as pd
from typing import Dict, Any
from src.strategies.base import BaseStrategy

class DonchianStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="Donchian_Channel", params=params)
    def get_default_params(self): return {"channel_period": 20}
    def get_param_space(self): return {"channel_period": ("int", 10, 50, 1)}
    
    def generate_signals(self, data):
        period = self.params["channel_period"]
        if len(data) < period: return pd.Series(0, index=data.index)
        df = data[['high','low','close']].copy()
        df['upper'] = df['high'].rolling(period).max()
        df['lower'] = df['low'].rolling(period).min()
        df['signal'] = 0
        df.loc[(df['close']>df['upper'].shift(1))&df['upper'].notna(), 'signal'] = 1
        df.loc[(df['close']<df['lower'].shift(1))&df['lower'].notna(), 'signal'] = -1
        return df['signal']
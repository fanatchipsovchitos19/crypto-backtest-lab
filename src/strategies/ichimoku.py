import pandas as pd
from typing import Dict, Any
from src.strategies.base import BaseStrategy

class IchimokuStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="Ichimoku", params=params)
    def get_default_params(self): return {"tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52}
    def get_param_space(self): return {"tenkan_period": ("int", 7, 20, 1), "kijun_period": ("int", 20, 40, 1), "senkou_b_period": ("int", 40, 80, 1)}
    
    def generate_signals(self, data):
        t, k, s = self.params["tenkan_period"], self.params["kijun_period"], self.params["senkou_b_period"]
        if len(data) < s+k: return pd.Series(0, index=data.index)
        df = data[['high','low','close']].copy()
        df['tenkan'] = (df['high'].rolling(t).max() + df['low'].rolling(t).min())/2
        df['kijun'] = (df['high'].rolling(k).max() + df['low'].rolling(k).min())/2
        df['senkou_a'] = ((df['tenkan']+df['kijun'])/2).shift(k)
        df['senkou_b'] = ((df['high'].rolling(s).max()+df['low'].rolling(s).min())/2).shift(k)
        df['cloud_top'] = df[['senkou_a','senkou_b']].max(axis=1)
        df['cloud_bottom'] = df[['senkou_a','senkou_b']].min(axis=1)
        df['signal'] = 0
        df.loc[(df['close']>df['cloud_top'])&(df['tenkan']>df['kijun'])&df['cloud_top'].notna(), 'signal'] = 1
        df.loc[(df['close']<df['cloud_bottom'])&(df['tenkan']<df['kijun'])&df['cloud_bottom'].notna(), 'signal'] = -1
        return df['signal']
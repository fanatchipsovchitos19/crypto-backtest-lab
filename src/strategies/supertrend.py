import pandas as pd
import numpy as np
from typing import Dict, Any
from src.strategies.base import BaseStrategy

class SupertrendStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="Supertrend", params=params)
    def get_default_params(self): return {"atr_period": 10, "multiplier": 3.0}
    def get_param_space(self): return {"atr_period": ("int", 7, 21, 1), "multiplier": ("float", 1.0, 5.0, 0.1)}
    
    @staticmethod
    def calculate_atr(data, period):
        high, low, close = data['high'], data['low'], data['close'].shift(1)
        tr = pd.concat([high-low, abs(high-close), abs(low-close)], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        for i in range(period, len(atr)): atr.iloc[i] = (atr.iloc[i-1]*(period-1)+tr.iloc[i])/period
        return atr
    
    def generate_signals(self, data):
        period = self.params["atr_period"]
        if len(data) < period+1: return pd.Series(0, index=data.index)
        df = data[['open','high','low','close']].copy()
        df['atr'] = self.calculate_atr(df, period)
        df['hl_avg'] = (df['high']+df['low'])/2
        mult = self.params["multiplier"]
        df['upper'] = df['hl_avg'] + mult*df['atr']
        df['lower'] = df['hl_avg'] - mult*df['atr']
        df['direction'] = 0
        first = df['atr'].first_valid_index()
        if first is None: return pd.Series(0, index=data.index)
        fi = df.index.get_loc(first)
        df.iloc[fi, df.columns.get_loc('direction')] = 1
        for i in range(fi+1, len(df)):
            prev_close, prev_dir = df['close'].iloc[i-1], df['direction'].iloc[i-1]
            if prev_dir == 1:
                df.iloc[i, df.columns.get_loc('direction')] = 1 if prev_close > df['lower'].iloc[i-1] else -1
            else:
                df.iloc[i, df.columns.get_loc('direction')] = -1 if prev_close < df['upper'].iloc[i-1] else 1
        df['signal'] = 0
        df.loc[(df['direction']==1)&(df['direction'].shift(1)==-1)&df['atr'].notna(), 'signal'] = 1
        df.loc[(df['direction']==-1)&(df['direction'].shift(1)==1)&df['atr'].notna(), 'signal'] = -1
        return df['signal']
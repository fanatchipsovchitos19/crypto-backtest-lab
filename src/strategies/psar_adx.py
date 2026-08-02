import pandas as pd
import numpy as np
from typing import Dict, Any
from src.strategies.base import BaseStrategy

class PSARADXStrategy(BaseStrategy):
    def __init__(self, params=None): super().__init__(name="PSAR_ADX", params=params)
    def get_default_params(self): return {"psar_step": 0.02, "psar_max": 0.2, "adx_period": 14, "adx_threshold": 25}
    def get_param_space(self): return {"psar_step": ("float", 0.01, 0.05, 0.005), "psar_max": ("float", 0.1, 0.5, 0.05), "adx_period": ("int", 10, 28, 1), "adx_threshold": ("int", 15, 40, 5)}
    
    @staticmethod
    def calculate_psar(data, step, max_step):
        high, low, n = data['high'].values, data['low'].values, len(data)
        psar = np.zeros(n); psar[0] = low[0]
        trend, ep, af = 1, high[0], step
        for i in range(1, n):
            psar[i] = psar[i-1] + af*(ep - psar[i-1])
            if trend == 1:
                psar[i] = min(psar[i], low[i-1], low[i-2] if i>=2 else low[i-1])
                if low[i] < psar[i]: trend, psar[i], ep, af = -1, ep, low[i], step
                elif high[i] > ep: ep, af = high[i], min(af+step, max_step)
            else:
                psar[i] = max(psar[i], high[i-1], high[i-2] if i>=2 else high[i-1])
                if high[i] > psar[i]: trend, psar[i], ep, af = 1, ep, high[i], step
                elif low[i] < ep: ep, af = low[i], min(af+step, max_step)
        return pd.Series(psar, index=data.index)
    
    @staticmethod
    def calculate_adx(data, period):
        high, low, close = data['high'], data['low'], data['close']
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        up_move, down_move = high-high.shift(1), low.shift(1)-low
        plus_dm = pd.Series(np.where((up_move>down_move)&(up_move>0), up_move, 0), index=data.index)
        minus_dm = pd.Series(np.where((down_move>up_move)&(down_move>0), down_move, 0), index=data.index)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100*plus_dm.ewm(alpha=1/period, adjust=False).mean()/atr
        minus_di = 100*minus_dm.ewm(alpha=1/period, adjust=False).mean()/atr
        dx = 100*abs(plus_di-minus_di)/(plus_di+minus_di+1e-10)
        return dx.ewm(alpha=1/period, adjust=False).mean()
    
    def generate_signals(self, data):
        if len(data) < self.params["adx_period"]*2: return pd.Series(0, index=data.index)
        df = data[['high','low','close']].copy()
        df['psar'] = self.calculate_psar(df, self.params["psar_step"], self.params["psar_max"])
        df['adx'] = self.calculate_adx(df, self.params["adx_period"])
        df['signal'] = 0
        df.loc[(df['close']>df['psar'])&(df['adx']>self.params["adx_threshold"])&df['psar'].notna()&df['adx'].notna(), 'signal'] = 1
        df.loc[(df['close']<df['psar'])&(df['adx']>self.params["adx_threshold"])&df['psar'].notna()&df['adx'].notna(), 'signal'] = -1
        return df['signal']
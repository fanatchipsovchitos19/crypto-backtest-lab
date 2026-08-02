import pandas as pd
import numpy as np
from typing import Dict, Any, Type, List, Optional
from src.strategies.base import BaseStrategy
from src.broker_simulator import SimulatedBroker
from src.optimizer import StrategyOptimizer
from src.metrics import calculate_all_metrics

class WalkForwardValidator:
    def __init__(self, strategy_class, data, broker, train_size=500, test_size=200, step_size=None, metric="sharpe_ratio", n_trials_per_window=50, symbol="BTCUSDT"):
        self.strategy_class = strategy_class
        self.data = data
        self.broker = broker
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.metric = metric
        self.n_trials_per_window = n_trials_per_window
        self.symbol = symbol
        self.windows = []
    
    def generate_windows(self):
        windows = []
        total_size = len(self.data)
        start_idx = 0
        window_num = 0
        while start_idx + self.train_size + self.test_size <= total_size:
            windows.append({'window_num':window_num,'train_start_idx':start_idx,'train_end_idx':start_idx+self.train_size-1,'test_start_idx':start_idx+self.train_size,'test_end_idx':min(start_idx+self.train_size+self.test_size-1,total_size-1),'train_start':self.data.index[start_idx],'train_end':self.data.index[start_idx+self.train_size-1],'test_start':self.data.index[start_idx+self.train_size],'test_end':self.data.index[min(start_idx+self.train_size+self.test_size-1,total_size-1)]})
            start_idx += self.step_size
            window_num += 1
        return windows
    
    def run(self, verbose=True):
        windows = self.generate_windows()
        if not windows: raise ValueError("Недостаточно данных для Walk-Forward.")
        in_sample_sharpes, out_of_sample_sharpes, all_best_params = [], [], []
        for window in windows:
            train_data = self.data.iloc[window['train_start_idx']:window['train_end_idx']+1]
            test_data = self.data.iloc[window['test_start_idx']:window['test_end_idx']+1]
            if len(train_data) < 50 or len(test_data) < 20: continue
            train_broker = SimulatedBroker(self.broker.initial_capital, self.broker.commission_percent, self.broker.slippage_percent)
            optimizer = StrategyOptimizer(self.strategy_class, train_data, train_broker, self.metric, symbol=self.symbol)
            opt_results = optimizer.optimize(n_trials=self.n_trials_per_window, sampler="tpe", verbose=False)
            best_params = opt_results['best_params']
            all_best_params.append(best_params)
            is_metrics = self._run_single(self.strategy_class(params=best_params), train_data, train_broker)
            if is_metrics: in_sample_sharpes.append(is_metrics.get('sharpe_ratio', 0))
            test_broker = SimulatedBroker(self.broker.initial_capital, self.broker.commission_percent, self.broker.slippage_percent)
            oos_metrics = self._run_single(self.strategy_class(params=best_params), test_data, test_broker)
            if oos_metrics: out_of_sample_sharpes.append(oos_metrics.get('sharpe_ratio', 0))
            if verbose and is_metrics and oos_metrics:
                print(f"  Окно {window['window_num']+1}: IS Sharpe={is_metrics.get('sharpe_ratio',0):.2f}, OOS Sharpe={oos_metrics.get('sharpe_ratio',0):.2f}, Params={best_params}")
            if oos_metrics:
                self.windows.append({'window_num':window['window_num'],'best_params':best_params,'is_metrics':is_metrics,'oos_metrics':oos_metrics,'test_start':window['test_start'],'test_end':window['test_end']})
        if not self.windows: raise ValueError("Ни одно окно не дало результатов.")
        avg_is = np.mean(in_sample_sharpes) if in_sample_sharpes else 0
        avg_oos = np.mean(out_of_sample_sharpes) if out_of_sample_sharpes else 0
        is_oos_ratio = avg_is/avg_oos if avg_oos > 0.01 else float('inf')
        stability = self._calc_stability(all_best_params)
        return {'n_windows':len(self.windows),'windows':self.windows,'avg_is_sharpe':avg_is,'avg_oos_sharpe':avg_oos,'is_oos_ratio':is_oos_ratio,'stability_score':stability,'all_best_params':all_best_params}
    
    def _run_single(self, strategy, data, broker):
        try:
            signals = strategy.generate_signals(data)
            broker.reset()
            for i, (ts, row) in enumerate(data.iterrows()):
                broker.execute_signal(ts, signals.iloc[i], {'open':row['open'],'high':row['high'],'low':row['low'],'close':row['close']}, self.symbol)
                broker.update_equity(ts, row['close'])
            eq = broker.get_equity_curve()
            tr = broker.get_trades_df()
            if len(eq) < 2 or tr.empty: return None
            return calculate_all_metrics(eq, tr, broker.initial_capital)
        except: return None
    
    def _calc_stability(self, all_params):
        if len(all_params) < 2: return 1.0
        scores = []
        for k in all_params[0].keys():
            vals = [p[k] for p in all_params]
            cv = np.std(vals)/abs(np.mean(vals)) if np.mean(vals) != 0 else 0
            scores.append(max(0, 1-cv))
        return np.mean(scores)
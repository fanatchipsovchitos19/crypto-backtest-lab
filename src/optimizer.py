import optuna
from typing import Dict, Any, Type
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.broker_simulator import SimulatedBroker
from src.metrics import calculate_all_metrics

class StrategyOptimizer:
    def __init__(self, strategy_class: Type[BaseStrategy], data: pd.DataFrame, broker: SimulatedBroker, metric: str = "sharpe_ratio", periods_per_year: int = 365*24, symbol: str = "BTCUSDT"):
        self.strategy_class = strategy_class
        self.data = data
        self.broker = broker
        self.metric = metric
        self.periods_per_year = periods_per_year
        self.symbol = symbol
        self.best_params = None
        self.best_metrics = None
        self.study = None
    
    def _objective(self, trial: optuna.Trial) -> float:
        strategy_instance = self.strategy_class()
        param_space = strategy_instance.get_param_space()
        suggested_params = {}
        for param_name, param_info in param_space.items():
            param_type = param_info[0]
            if param_type == "int": suggested_params[param_name] = trial.suggest_int(param_name, param_info[1], param_info[2], step=param_info[3] if len(param_info)>3 else 1)
            elif param_type == "float": suggested_params[param_name] = trial.suggest_float(param_name, param_info[1], param_info[2], step=param_info[3] if len(param_info)>3 else None)
            elif param_type == "categorical": suggested_params[param_name] = trial.suggest_categorical(param_name, param_info[1])
        if 'fast_period' in suggested_params and 'slow_period' in suggested_params:
            if suggested_params['fast_period'] >= suggested_params['slow_period']: return float('inf')
        strategy = self.strategy_class(params=suggested_params)
        signals = strategy.generate_signals(self.data)
        self.broker.reset()
        for i, (timestamp, row) in enumerate(self.data.iterrows()):
            signal = signals.iloc[i]
            price_data = {'open':row['open'],'high':row['high'],'low':row['low'],'close':row['close']}
            self.broker.execute_signal(timestamp, signal, price_data, self.symbol)
            self.broker.update_equity(timestamp, row['close'])
        equity_curve = self.broker.get_equity_curve()
        trades_df = self.broker.get_trades_df()
        if len(equity_curve) < 2 or trades_df.empty: return float('inf')
        metrics = calculate_all_metrics(equity_curve, trades_df, self.broker.initial_capital, self.periods_per_year)
        target_value = metrics.get(self.metric, 0)
        if self.metric in ('max_drawdown_pct',): return abs(target_value)
        else: return -target_value
    
    def optimize(self, n_trials=100, sampler="tpe", n_jobs=1, verbose=True):
        if sampler == "cmaes": optuna_sampler = optuna.samplers.CmaEsSampler()
        elif sampler == "random": optuna_sampler = optuna.samplers.RandomSampler()
        else: optuna_sampler = optuna.samplers.TPESampler()
        self.study = optuna.create_study(sampler=optuna_sampler, direction="minimize")
        optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)
        self.study.optimize(self._objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)
        self.best_params = self.study.best_params
        self.best_metrics = -self.study.best_value
        return self.get_results()
    
    def get_results(self):
        if self.study is None: return {}
        return {'best_params':self.study.best_params,'best_metric_value':-self.study.best_value,'metric_name':self.metric,'n_trials':len(self.study.trials),'study':self.study}
    
    def get_top_trials(self, n=5):
        if self.study is None: return pd.DataFrame()
        trials = []
        for trial in self.study.trials:
            if trial.value is not None and trial.value != float('inf'):
                trials.append({**trial.params, 'metric_value':-trial.value, 'trial_number':trial.number})
        return pd.DataFrame(trials).sort_values('metric_value', ascending=False).head(n)
import pandas as pd
from typing import Dict, Any, Optional
from src.strategies.base import BaseStrategy
from src.broker_simulator import SimulatedBroker
from src.metrics import calculate_all_metrics

class BacktestEngine:
    def __init__(self, strategy: BaseStrategy, broker: SimulatedBroker, symbol: str = "BTCUSDT"):
        self.strategy = strategy
        self.broker = broker
        self.symbol = symbol
        self.signals: Optional[pd.Series] = None
        self.equity_curve: Optional[pd.Series] = None
        self.trades_df: Optional[pd.DataFrame] = None
        self.metrics: Optional[Dict[str, float]] = None
    
    def run(self, data: pd.DataFrame, verbose: bool = False) -> Dict[str, Any]:
        self.signals = self.strategy.generate_signals(data)
        if verbose:
            buys = (self.signals == 1).sum()
            sells = (self.signals == -1).sum()
            print(f"  Сигналов: BUY={buys}, SELL={sells}")
        self.broker.reset()
        for i, (timestamp, row) in enumerate(data.iterrows()):
            signal = self.signals.iloc[i]
            price_data = {'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close']}
            self.broker.execute_signal(timestamp, signal, price_data, self.symbol)
            self.broker.update_equity(timestamp, row['close'])
        self.equity_curve = self.broker.get_equity_curve()
        self.trades_df = self.broker.get_trades_df()
        self.metrics = calculate_all_metrics(self.equity_curve, self.trades_df, self.broker.initial_capital)
        return {'signals': self.signals, 'equity_curve': self.equity_curve, 'trades_df': self.trades_df, 'metrics': self.metrics}
    
    def get_summary(self) -> Dict[str, Any]:
        if self.metrics is None: raise RuntimeError("Сначала запустите run()")
        return {'total_return_pct': self.metrics.get('total_return_pct', 0), 'sharpe_ratio': self.metrics.get('sharpe_ratio', 0), 'max_drawdown_pct': self.metrics.get('max_drawdown_pct', 0), 'win_rate': self.metrics.get('win_rate', 0), 'profit_factor': self.metrics.get('profit_factor', 0), 'total_trades': self.metrics.get('total_trades', 0)}
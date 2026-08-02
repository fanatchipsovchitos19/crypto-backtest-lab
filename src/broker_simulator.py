from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import pandas as pd
import numpy as np

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Trade:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    commission_asset: str = "USDT"

@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

class SimulatedBroker:
    def __init__(self, initial_capital=10_000, commission_percent=0.001, slippage_percent=0.0005, slippage_model="fixed"):
        self.initial_capital = initial_capital
        self.commission_percent = commission_percent
        self.slippage_percent = slippage_percent
        self.slippage_model = slippage_model
        self.cash = initial_capital
        self.position = Position(symbol="")
        self.trades: List[Trade] = []
        self.equity_history: Dict[pd.Timestamp, float] = {}
    
    def reset(self):
        self.cash = self.initial_capital
        self.position = Position(symbol="")
        self.trades = []
        self.equity_history = {}
    
    def get_equity(self, current_price):
        return self.cash + self.position.quantity * current_price
    
    def execute_signal(self, timestamp, signal, price_data, symbol):
        if signal == 0: return None
        close_price = price_data['close']
        if signal == 1: return self._execute_buy(timestamp, symbol, close_price)
        elif signal == -1: return self._execute_sell(timestamp, symbol, close_price)
        return None
    
    def _execute_buy(self, timestamp, symbol, price):
        if self.position.quantity > 0: return None
        execution_price = self._apply_slippage(price, OrderSide.BUY)
        max_spend = self.cash / (1 + self.commission_percent)
        quantity = max_spend / execution_price
        if quantity <= 0: return None
        commission = quantity * execution_price * self.commission_percent
        self.cash -= (quantity * execution_price + commission)
        self.position.quantity = quantity
        self.position.avg_entry_price = execution_price
        trade = Trade(timestamp=timestamp, symbol=symbol, side=OrderSide.BUY, quantity=quantity, price=execution_price, commission=commission)
        self.trades.append(trade)
        return trade
    
    def _execute_sell(self, timestamp, symbol, price):
        if self.position.quantity <= 0: return None
        execution_price = self._apply_slippage(price, OrderSide.SELL)
        quantity = self.position.quantity
        gross_proceeds = quantity * execution_price
        commission = gross_proceeds * self.commission_percent
        entry_value = quantity * self.position.avg_entry_price
        pnl = gross_proceeds - entry_value - commission
        self.cash += gross_proceeds - commission
        self.position.realized_pnl += pnl
        self.position.quantity = 0.0
        self.position.avg_entry_price = 0.0
        trade = Trade(timestamp=timestamp, symbol=symbol, side=OrderSide.SELL, quantity=quantity, price=execution_price, commission=commission)
        self.trades.append(trade)
        return trade
    
    def _apply_slippage(self, price, side):
        if self.slippage_model == "fixed": slippage = self.slippage_percent
        else: slippage = abs(np.random.normal(self.slippage_percent, self.slippage_percent / 2))
        if side == OrderSide.BUY: return price * (1 + slippage)
        else: return price * (1 - slippage)
    
    def update_equity(self, timestamp, current_price):
        self.equity_history[timestamp] = self.get_equity(current_price)
    
    def get_trades_df(self):
        if not self.trades: return pd.DataFrame()
        return pd.DataFrame([{'timestamp':t.timestamp,'symbol':t.symbol,'side':t.side.value,'quantity':t.quantity,'price':t.price,'commission':t.commission} for t in self.trades])
    
    def get_equity_curve(self):
        return pd.Series(self.equity_history, name='equity')
    
    def get_statistics(self):
        if not self.trades: return {'total_trades':0,'total_pnl':0.0,'return_percent':0.0}
        if len(self.equity_history) > 0:
            eq = self.get_equity_curve()
            final_equity = eq.iloc[-1]
            total_return = (final_equity - self.initial_capital) / self.initial_capital
        else: total_return = 0.0
        trades_df = self.get_trades_df()
        return {'total_trades':len(self.trades), 'buy_trades':len(trades_df[trades_df['side']=='buy']), 'sell_trades':len(trades_df[trades_df['side']=='sell']), 'total_pnl':self.position.realized_pnl, 'return_percent':total_return*100}
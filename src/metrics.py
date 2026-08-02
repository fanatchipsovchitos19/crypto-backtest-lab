import pandas as pd
import numpy as np
from typing import Dict

def calculate_returns(equity_curve): return equity_curve.pct_change().dropna()

def annualize_sharpe(returns, risk_free_rate=0.02, periods_per_year=365*24):
    if len(returns) < 2: return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    mean_ex, std_ex = excess.mean(), excess.std()
    if std_ex == 0: return 0.0
    return (mean_ex / std_ex) * np.sqrt(periods_per_year)

def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=365*24):
    if len(returns) < 2: return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    if len(downside) < 2: return 0.0 if excess.mean() <= 0 else np.inf
    d_std = downside.std()
    if d_std == 0: return 0.0 if excess.mean() <= 0 else np.inf
    return (excess.mean() / d_std) * np.sqrt(periods_per_year)

def max_drawdown(equity_curve):
    if len(equity_curve) < 2: return {'max_drawdown_pct':0.0,'max_drawdown_duration':0,'max_drawdown_start':None,'max_drawdown_end':None,'current_drawdown':0.0}
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    peak_idx = rolling_max[:max_dd_idx].idxmax()
    recovery_mask = (equity_curve[max_dd_idx:] >= rolling_max[max_dd_idx])
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        duration = len(equity_curve[peak_idx:recovery_idx])
    else:
        recovery_idx = equity_curve.index[-1]
        duration = len(equity_curve[peak_idx:])
    return {'max_drawdown_pct':max_dd*100,'max_drawdown_duration':duration,'max_drawdown_start':peak_idx,'max_drawdown_end':recovery_idx,'current_drawdown':drawdown.iloc[-1]*100}

def calmar_ratio(equity_curve, periods_per_year=365*24):
    if len(equity_curve) < 2: return 0.0
    dd_info = max_drawdown(equity_curve)
    max_dd = abs(dd_info['max_drawdown_pct'])
    if max_dd < 0.01: return 0.0
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year
    annual_return = total_return if years < 0.01 else (1 + total_return) ** (1 / years) - 1
    return (annual_return * 100) / max_dd

def win_rate_metrics(trades_df):
    empty = {'win_rate':0.0,'avg_win':0.0,'avg_loss':0.0,'profit_factor':0.0,'total_pnl':0.0,'total_trades':0,'winning_trades':0,'losing_trades':0}
    if trades_df.empty: return empty
    buy_trades = trades_df[trades_df['side']=='buy'].reset_index(drop=True)
    sell_trades = trades_df[trades_df['side']=='sell'].reset_index(drop=True)
    if len(buy_trades)==0 or len(sell_trades)==0: return empty
    pnl_list = []
    for i in range(min(len(buy_trades), len(sell_trades))):
        bp, sp, qty = buy_trades.iloc[i]['price'], sell_trades.iloc[i]['price'], buy_trades.iloc[i]['quantity']
        pnl = (sp - bp) * qty - (bp*qty + sp*qty)*0.001
        pnl_list.append(pnl)
    if not pnl_list: return empty
    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p <= 0]
    wr = len(wins)/len(pnl_list)*100
    avg_w = np.mean(wins) if wins else 0.0
    avg_l = np.mean(losses) if losses else 0.0
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = gp/gl if gl > 0 else float('inf')
    return {'win_rate':wr,'avg_win':avg_w,'avg_loss':avg_l,'profit_factor':pf,'total_pnl':sum(pnl_list),'total_trades':len(pnl_list),'winning_trades':len(wins),'losing_trades':len(losses)}

def calculate_all_metrics(equity_curve, trades_df, initial_capital, periods_per_year=365*24, risk_free_rate=0.02):
    returns = calculate_returns(equity_curve)
    dd_info = max_drawdown(equity_curve)
    trade_metrics = win_rate_metrics(trades_df)
    total_return_pct = ((equity_curve.iloc[-1] - initial_capital) / initial_capital) * 100
    return {
        'total_return_pct':total_return_pct,'total_pnl':equity_curve.iloc[-1]-initial_capital,
        'final_equity':equity_curve.iloc[-1],'initial_capital':initial_capital,
        'sharpe_ratio':annualize_sharpe(returns,risk_free_rate,periods_per_year),
        'sortino_ratio':sortino_ratio(returns,risk_free_rate,periods_per_year),
        'calmar_ratio':calmar_ratio(equity_curve,periods_per_year),
        'max_drawdown_pct':dd_info['max_drawdown_pct'],'max_drawdown_duration':dd_info['max_drawdown_duration'],
        'current_drawdown_pct':dd_info['current_drawdown'],
        'annual_volatility':returns.std()*np.sqrt(periods_per_year)*100,
        'daily_volatility':returns.std()*np.sqrt(24)*100,
        'win_rate':trade_metrics['win_rate'],'avg_win':trade_metrics['avg_win'],
        'avg_loss':trade_metrics['avg_loss'],'profit_factor':trade_metrics['profit_factor'],
        'total_trades':trade_metrics['total_trades'],'winning_trades':trade_metrics['winning_trades'],
        'losing_trades':trade_metrics['losing_trades'],
    }
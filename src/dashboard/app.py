"""
Crypto Backtest Lab — Streamlit Dashboard
"""
import sys
from pathlib import Path

# Добавляем корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.data_loader import BinanceDataLoader
from src.broker_simulator import SimulatedBroker
from src.backtest_engine import BacktestEngine
from src.strategies import AVAILABLE_STRATEGIES
from src.optimizer import StrategyOptimizer
from src.metrics import calculate_all_metrics, print_metrics_report


# ── Конфигурация страницы ────────────────────────────────
st.set_page_config(
    page_title="Crypto Backtest Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Заголовок ────────────────────────────────────────────
st.title("📊 Crypto Backtest Lab")
st.markdown("*Профессиональный бэктестинг крипто-стратегий с AI-оптимизатором*")

# ── Боковая панель: параметры ────────────────────────────
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор стратегии
    strategy_names = list(AVAILABLE_STRATEGIES.keys())
    strategy_display = {
        'sma_crossover': 'SMA Crossover',
        'ema_ribbon': 'EMA Ribbon',
        'rsi_mean_reversion': 'RSI Mean Reversion',
        'bollinger_bands': 'Bollinger Bands',
        'macd_classic': 'MACD Classic',
        'supertrend': 'Supertrend',
    }
    selected_strategy_key = st.selectbox(
        "Стратегия",
        strategy_names,
        format_func=lambda x: strategy_display.get(x, x),
    )
    
    # Данные
    st.subheader("📡 Данные")
    symbol = st.text_input("Торговая пара", value="BTCUSDT")
    interval = st.selectbox("Интервал", ["1h", "4h", "1d", "15m", "30m"], index=0)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Начало", value=datetime(2024, 9, 1))
    with col2:
        end_date = st.date_input("Конец", value=datetime(2024, 12, 1))
    
    # Брокер
    st.subheader("💰 Брокер")
    initial_capital = st.number_input("Начальный капитал ($)", value=10_000, min_value=100, step=1000)
    commission = st.slider("Комиссия (%)", 0.0, 1.0, 0.1, 0.01) / 100
    slippage = st.slider("Проскальзывание (%)", 0.0, 1.0, 0.05, 0.01) / 100
    
    # Кнопка запуска
    st.markdown("---")
    run_backtest = st.button("🚀 Запустить бэктест", type="primary", use_container_width=True)
    
    # Оптимизация
    st.markdown("---")
    st.subheader("🤖 AI-Оптимизатор")
    enable_optimization = st.checkbox("Оптимизировать параметры", value=False)
    n_trials = st.slider("Количество попыток", 10, 200, 50, 10, disabled=not enable_optimization)
    metric_opt = st.selectbox("Метрика для оптимизации", 
                               ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "total_return_pct"],
                               disabled=not enable_optimization)


# ── Загрузка данных ─────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(symbol, interval, start, end):
    loader = BinanceDataLoader()
    return loader.fetch_ohlcv(symbol, interval, start=str(start), end=str(end))


# ── Запуск бэктеста ─────────────────────────────────────
if run_backtest:
    with st.spinner("Загрузка данных..."):
        try:
            data = load_data(symbol, interval, start_date, end_date)
        except Exception as e:
            st.error(f"Ошибка загрузки данных: {e}")
            st.stop()
    
    if len(data) < 50:
        st.error("Недостаточно данных. Увеличьте период.")
        st.stop()
    
    st.success(f"Загружено {len(data):,} свечей ({data.index[0]} → {data.index[-1]})")
    
    # Создаём стратегию и брокера
    strategy_class = AVAILABLE_STRATEGIES[selected_strategy_key]
    strategy = strategy_class()
    broker = SimulatedBroker(
        initial_capital=initial_capital,
        commission_percent=commission,
        slippage_percent=slippage,
    )
    
    # ── Оптимизация ─────────────────────────────────────
    if enable_optimization:
        with st.spinner(f"Оптимизация параметров ({n_trials} попыток)..."):
            optimizer = StrategyOptimizer(
                strategy_class=strategy_class,
                data=data,
                broker=broker,
                metric=metric_opt,
            )
            opt_results = optimizer.optimize(n_trials=n_trials, sampler="tpe", verbose=False)
            
            st.success(f"Оптимизация завершена! Лучшее значение {metric_opt}: {opt_results['best_metric_value']:.4f}")
            
            with st.expander("📊 Результаты оптимизации"):
                st.json(opt_results['best_params'])
                
                # График сходимости
                if opt_results['study']:
                    fig_opt = optuna_plot(opt_results['study'])
                    st.plotly_chart(fig_opt, use_container_width=True)
            
            strategy.set_params(opt_results['best_params'])
    
    # ── Запуск бэктеста ─────────────────────────────────
    with st.spinner("Бэктест..."):
        engine = BacktestEngine(strategy, broker)
        results = engine.run(data)
    
    # ── Отображение результатов ─────────────────────────
    st.markdown("---")
    
    # Ключевые метрики в ряд
    m = engine.metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_color = "normal" if m['total_return_pct'] >= 0 else "inverse"
        st.metric("Доходность", f"{m['total_return_pct']:.2f}%", 
                  delta=f"${m['total_pnl']:,.0f}", delta_color=delta_color)
    
    with col2:
        st.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%")
    
    with col4:
        st.metric("Win Rate", f"{m['win_rate']:.1f}%")
    
    with col5:
        st.metric("Сделок", f"{m['total_trades']}")
    
    # Графики
    st.markdown("---")
    st.subheader("📈 Графики")
    
    # Создаём интерактивные графики Plotly
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Цена + Сигналы", "Кривая эквити", "Просадка"),
    )
    
    # График 1: Цена + сигналы
    fig.add_trace(
        go.Scatter(x=data.index, y=data['close'], mode='lines', 
                   name=f'{symbol}', line=dict(color='gray', width=0.8)),
        row=1, col=1,
    )
    
    # Сигналы BUY
    buy_idx = data.index[results['signals'] == 1]
    if len(buy_idx) > 0:
        fig.add_trace(
            go.Scatter(x=buy_idx, y=data.loc[buy_idx, 'low'] * 0.998,
                       mode='markers', name='BUY', 
                       marker=dict(symbol='triangle-up', size=10, color='green')),
            row=1, col=1,
        )
    
    # Сигналы SELL
    sell_idx = data.index[results['signals'] == -1]
    if len(sell_idx) > 0:
        fig.add_trace(
            go.Scatter(x=sell_idx, y=data.loc[sell_idx, 'high'] * 1.002,
                       mode='markers', name='SELL',
                       marker=dict(symbol='triangle-down', size=10, color='red')),
            row=1, col=1,
        )
    
    # График 2: Эквити
    equity = results['equity_curve']
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, mode='lines',
                   name='Equity', line=dict(color='green', width=1.5),
                   fill='tozeroy', fillcolor='rgba(0,255,0,0.05)'),
        row=2, col=1,
    )
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", 
                  annotation_text="Initial", row=2, col=1)
    
    # График 3: Просадка
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, mode='lines',
                   name='Drawdown', line=dict(color='red', width=1),
                   fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'),
        row=3, col=1,
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        title_text=f"{strategy_display.get(selected_strategy_key, selected_strategy_key)} — {symbol} {interval}",
    )
    fig.update_xaxes(title_text="Дата", row=3, col=1)
    fig.update_yaxes(title_text="Цена", row=1, col=1)
    fig.update_yaxes(title_text="Эквити ($)", row=2, col=1)
    fig.update_yaxes(title_text="Просадка (%)", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Детальные метрики в двух колонках
    st.markdown("---")
    st.subheader("📊 Детальные метрики")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Доходность и риск**")
        metrics_df = pd.DataFrame({
            'Метрика': ['Доходность', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                        'Max Drawdown', 'Годовая волатильность', 'Дневная волатильность'],
            'Значение': [
                f"{m['total_return_pct']:.2f}%",
                f"{m['sharpe_ratio']:.2f}",
                f"{m['sortino_ratio']:.2f}",
                f"{m['calmar_ratio']:.2f}",
                f"{m['max_drawdown_pct']:.2f}%",
                f"{m['annual_volatility']:.2f}%",
                f"{m['daily_volatility']:.2f}%",
            ]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Торговля**")
        trade_df = pd.DataFrame({
            'Метрика': ['Всего сделок', 'Выигрышных', 'Проигрышных', 'Win Rate',
                        'Средний выигрыш', 'Средний проигрыш', 'Profit Factor'],
            'Значение': [
                f"{m['total_trades']}",
                f"{m['winning_trades']}",
                f"{m['losing_trades']}",
                f"{m['win_rate']:.1f}%",
                f"${m['avg_win']:,.2f}",
                f"${m['avg_loss']:,.2f}",
                f"{m['profit_factor']:.2f}",
            ]
        })
        st.dataframe(trade_df, hide_index=True, use_container_width=True)
    
    # Список сделок
    if not results['trades_df'].empty:
        with st.expander("📋 Список сделок"):
            st.dataframe(results['trades_df'], use_container_width=True)
    
    # Оценка стратегии
    st.markdown("---")
    st.subheader("🏆 Оценка стратегии")
    
    score = 0
    checks = []
    
    if m['sharpe_ratio'] > 1.0:
        score += 2
        checks.append("✅ Sharpe > 1.0 — хорошо")
    elif m['sharpe_ratio'] > 0.5:
        score += 1
        checks.append("⚠️ Sharpe 0.5–1.0 — средне")
    else:
        checks.append("❌ Sharpe < 0.5 — плохо")
    
    if m['profit_factor'] > 1.5:
        score += 2
        checks.append("✅ Profit Factor > 1.5 — отлично")
    elif m['profit_factor'] > 1.0:
        score += 1
        checks.append("⚠️ Profit Factor 1.0–1.5 — ок")
    else:
        checks.append("❌ Profit Factor < 1.0 — убыточно")
    
    if abs(m['max_drawdown_pct']) < 20:
        score += 2
        checks.append("✅ Max DD < 20% — приемлемо")
    elif abs(m['max_drawdown_pct']) < 30:
        score += 1
        checks.append("⚠️ Max DD 20–30% — высоковато")
    else:
        checks.append("❌ Max DD > 30% — опасно")
    
    if m['win_rate'] > 40:
        score += 1
        checks.append("✅ Win Rate > 40% — ок")
    else:
        checks.append("⚠️ Win Rate < 40% — низкий")
    
    for check in checks:
        st.text(check)
    
    progress = score / 7
    st.progress(progress, text=f"Общий счёт: {score}/7")
    
    if score >= 6:
        st.success("🟢 СТРАТЕГИЯ ГОТОВА К ТОРГАМ")
    elif score >= 4:
        st.warning("🟡 ТРЕБУЕТСЯ ДОРАБОТКА")
    else:
        st.error("🔴 НЕ РЕКОМЕНДУЕТСЯ К ТОРГАМ")


def optuna_plot(study):
    """График сходимости Optuna."""
    trials = [t for t in study.trials if t.value is not None and t.value != float('inf')]
    values = [-t.value for t in trials]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values, mode='lines', name='Значение метрики',
        line=dict(color='blue', width=1)
    ))
    fig.add_trace(go.Scatter(
        y=[max(values[:i+1]) for i in range(len(values))],
        mode='lines', name='Лучшее значение',
        line=dict(color='green', width=2)
    ))
    fig.update_layout(
        title="Сходимость оптимизации",
        xaxis_title="Попытка",
        yaxis_title="Значение метрики",
        height=300,
        showlegend=True,
    )
    return fig


# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Crypto Backtest Lab v1.0 | "
    "<a href='https://github.com' target='_blank'>GitHub</a> | "
    "AI-оптимизатор на базе Optuna</p>",
    unsafe_allow_html=True,
)
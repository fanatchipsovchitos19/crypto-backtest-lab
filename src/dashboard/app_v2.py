"""
Crypto Backtest Lab — Streamlit Dashboard
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from src.data_loader import BinanceDataLoader
from src.broker_simulator import SimulatedBroker
from src.backtest_engine import BacktestEngine
from src.strategies import AVAILABLE_STRATEGIES
from src.optimizer import StrategyOptimizer

st.set_page_config(page_title="Crypto Backtest Lab", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

if 'results' not in st.session_state: st.session_state.results = None
if 'metrics' not in st.session_state: st.session_state.metrics = None
if 'data' not in st.session_state: st.session_state.data = None
if 'strategy' not in st.session_state: st.session_state.strategy = None
if 'pdf_data' not in st.session_state: st.session_state.pdf_data = None
if 'pdf_filename' not in st.session_state: st.session_state.pdf_filename = None
if 'api_key' not in st.session_state: st.session_state.api_key = ''

st.title("📊 Crypto Backtest Lab")
st.markdown("*Профессиональный бэктестинг крипто-стратегий с оптимизатором*")

with st.sidebar:
    st.header("⚙️ Настройки")
    
    strategy_names = list(AVAILABLE_STRATEGIES.keys())
    strategy_display = {
        'sma_crossover': 'SMA Crossover', 'ema_ribbon': 'EMA Ribbon',
        'rsi_mean_reversion': 'RSI Mean Reversion', 'bollinger_bands': 'Bollinger Bands',
        'macd_classic': 'MACD Classic', 'supertrend': 'Supertrend',
        'ichimoku': 'Ichimoku Cloud', 'donchian': 'Donchian Channel',
        'psar_adx': 'PSAR + ADX', 'xgboost_ml': 'XGBoost ML',
    }
    selected_strategy_key = st.selectbox("Стратегия", strategy_names, format_func=lambda x: strategy_display.get(x, x))
    
    st.subheader("📡 Данные")
    symbol = st.text_input("Торговая пара", value="BTCUSDT").upper()
    interval = st.selectbox("Интервал", ["1h", "4h", "1d", "15m", "30m"], index=0)
    
    col1, col2 = st.columns(2)
    with col1: start_date = st.date_input("Начало", value=datetime(2024, 9, 1))
    with col2: end_date = st.date_input("Конец", value=datetime(2024, 12, 1))
    
    st.subheader("💰 Брокер")
    initial_capital = st.number_input("Капитал ($)", value=10_000, min_value=100, step=1000)
    commission = st.slider("Комиссия (%)", 0.0, 1.0, 0.1, 0.01) / 100
    slippage = st.slider("Проскальзывание (%)", 0.0, 1.0, 0.05, 0.01) / 100
    
    st.markdown("---")
    st.subheader("🔧 Оптимизатор")
    enable_optimization = st.checkbox("Оптимизировать параметры", value=False)
    n_trials = st.slider("Попыток", 10, 200, 50, 10, disabled=not enable_optimization)
    metric_opt = st.selectbox("Целевая метрика", ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "total_return_pct"],
                              format_func=lambda x: {'sharpe_ratio':'Sharpe Ratio','sortino_ratio':'Sortino Ratio','calmar_ratio':'Calmar Ratio','total_return_pct':'Доходность (%)'}.get(x,x),
                              disabled=not enable_optimization)
    
    st.markdown("---")
    run_backtest = st.button("🚀 Запустить бэктест", type="primary", use_container_width=True)
    
# ── Запуск бэктеста ──
if run_backtest:
    
    with st.spinner(f"Загрузка данных {symbol} {interval}..."):
        try:
            loader = BinanceDataLoader()
            data = loader.fetch_ohlcv(symbol, interval, str(start_date), str(end_date))
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
            st.stop()
    
    if len(data) < 50:
        st.error("Недостаточно данных.")
        st.stop()
    
    st.session_state.data = data
    st.session_state.symbol = symbol
    st.session_state.interval = interval
    st.session_state.strategy_key = selected_strategy_key
    st.session_state.opt_enabled = enable_optimization
    
    strategy_class = AVAILABLE_STRATEGIES[selected_strategy_key]
    strategy = strategy_class()
    broker = SimulatedBroker(initial_capital=initial_capital, commission_percent=commission, slippage_percent=slippage)
    
    if enable_optimization:
        with st.spinner(f"Оптимизация {n_trials} попыток..."):
            optimizer = StrategyOptimizer(
                strategy_class=strategy_class, data=data, broker=broker,
                metric=metric_opt, symbol=symbol,
            )
            opt_results = optimizer.optimize(n_trials=n_trials, sampler="tpe", verbose=False)
            strategy.set_params(opt_results['best_params'])
    
    with st.spinner("Бэктест..."):
        broker.reset()
        engine = BacktestEngine(strategy, broker, symbol=symbol)
        results = engine.run(data)
    
    st.session_state.results = results
    st.session_state.metrics = engine.metrics
    st.session_state.strategy = strategy


# ── Отображение результатов ──
if st.session_state.results is not None:
    
    m = st.session_state.metrics
    results = st.session_state.results
    data = st.session_state.data
    total_cycles = m['winning_trades'] + m['losing_trades']
    
    st.success(f"Загружено {len(data):,} свечей | {data.index[0].strftime('%d.%m.%Y')} → {data.index[-1].strftime('%d.%m.%Y')}")
    
    # ── Ключевые метрики ──
    st.markdown("---")
    st.subheader("📊 Ключевые метрики")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        color = "normal" if m['total_return_pct'] >= 0 else "inverse"
        st.metric("Доходность", f"{m['total_return_pct']:.2f}%", delta=f"${m['total_pnl']:,.0f}", delta_color=color)
    with col2: st.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
    with col3: st.metric("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%")
    with col4: st.metric("Win Rate", f"{m['win_rate']:.1f}%")
    with col5: st.metric("Сделок", total_cycles)
    
    # ── Интерактивные графики ──
    st.markdown("---")
    st.subheader("📈 Интерактивные графики")
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{st.session_state.symbol} — Цена и сигналы", "Кривая эквити", "Просадка (%)"),
    )
    
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines',
                   name=st.session_state.symbol, line=dict(color='#636efa', width=1)), row=1, col=1)
    
    buy_idx = data.index[results['signals'] == 1]
    sell_idx = data.index[results['signals'] == -1]
    
    if len(buy_idx) > 0:
        fig.add_trace(go.Scatter(x=buy_idx, y=data.loc[buy_idx, 'low'] * 0.998, mode='markers',
                       name='BUY', marker=dict(symbol='triangle-up', size=10, color='#00cc96')), row=1, col=1)
    if len(sell_idx) > 0:
        fig.add_trace(go.Scatter(x=sell_idx, y=data.loc[sell_idx, 'high'] * 1.002, mode='markers',
                       name='SELL', marker=dict(symbol='triangle-down', size=10, color='#ef553b')), row=1, col=1)
    
    equity = results['equity_curve']
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode='lines', name='Equity',
                   line=dict(color='#00cc96', width=1.5), fill='tozeroy', fillcolor='rgba(0,204,150,0.08)'), row=2, col=1)
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", annotation_text="Initial", row=2, col=1)
    
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, mode='lines', name='Drawdown',
                   line=dict(color='#ef553b', width=1), fill='tozeroy', fillcolor='rgba(239,85,59,0.1)'), row=3, col=1)
    
    fig.update_layout(height=750, showlegend=True, hovermode='x unified',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="", row=3, col=1)
    fig.update_yaxes(title_text="Цена", row=1, col=1)
    fig.update_yaxes(title_text="Эквити ($)", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Детальные метрики ──
    st.markdown("---")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📉 Риск-метрики")
        st.markdown(f"""
        | Метрика | Значение |
        |---------|----------|
        | Sharpe Ratio | {m['sharpe_ratio']:.2f} |
        | Sortino Ratio | {m['sortino_ratio']:.2f} |
        | Calmar Ratio | {m['calmar_ratio']:.2f} |
        | Макс. просадка | {m['max_drawdown_pct']:.2f}% |
        | Волатильность (год) | {m['annual_volatility']:.2f}% |
        """)
    
    with col_right:
        st.subheader("📈 Торговые метрики")
        st.markdown(f"""
        | Метрика | Значение |
        |---------|----------|
        | Сделок | {total_cycles} |
        | Win Rate | {m['win_rate']:.1f}% |
        | Ср. выигрыш | ${m['avg_win']:,.2f} |
        | Ср. проигрыш | ${m['avg_loss']:,.2f} |
        | Profit Factor | {m['profit_factor']:.2f} |
        """)
            # ── Оценка стратегии ──
    st.markdown("---")
    st.subheader("🏆 Оценка стратегии")
    
    score = 0
    max_score = 10
    checks = []
    
    ret_score = 0
    if m['total_return_pct'] > 100: ret_score = 2
    elif m['total_return_pct'] > 30: ret_score = 1
    elif m['total_return_pct'] > 0: ret_score = 0
    else: ret_score = -2
    score += ret_score
    icon = "✅" if ret_score >= 2 else "⚠️" if ret_score >= 0 else "❌"
    checks.append((icon, f"Доходность {m['total_return_pct']:.1f}%", ret_score, 2))
    
    trade_score = 2 if total_cycles >= 100 else 1 if total_cycles >= 30 else 0
    score += trade_score
    checks.append(("✅" if trade_score >= 2 else "⚠️", f"Сделок: {total_cycles}", trade_score, 2))
    
    sharpe_score = 2 if m['sharpe_ratio'] > 1.5 else 1 if m['sharpe_ratio'] > 0.7 else 0
    score += sharpe_score
    checks.append(("✅" if sharpe_score >= 2 else "⚠️" if sharpe_score >= 1 else "❌", f"Sharpe {m['sharpe_ratio']:.2f}", sharpe_score, 2))
    
    dd_score = 2 if abs(m['max_drawdown_pct']) < 20 else 1 if abs(m['max_drawdown_pct']) < 40 else 0
    score += dd_score
    checks.append(("✅" if dd_score >= 2 else "⚠️" if dd_score >= 1 else "❌", f"Max DD {m['max_drawdown_pct']:.1f}%", dd_score, 2))
    
    wr_score = 2 if m['win_rate'] > 55 else 1 if m['win_rate'] >= 40 else 0
    score += wr_score
    checks.append(("✅" if wr_score >= 2 else "⚠️" if wr_score >= 1 else "⚠️", f"Win Rate {m['win_rate']:.1f}%", wr_score, 2))
    
    for icon, text, pts, max_pts in checks:
        st.text(f"{icon} {text}  ({pts}/{max_pts} балла)")
    
    score = max(0, score)
    st.progress(score / max_score, text=f"Общий счёт: {score}/{max_score}")
    
    if score >= 8: st.success("🟢 СТРАТЕГИЯ ГОТОВА К РЕАЛЬНОЙ ТОРГОВЛЕ")
    elif score >= 6: st.success("🟢 СТРАТЕГИЯ ПРИБЫЛЬНА, НО ТРЕБУЕТ УПРАВЛЕНИЯ РИСКАМИ")
    elif score >= 3: st.warning("🟡 СТРАТЕГИЯ ТРЕБУЕТ ДОРАБОТКИ")
    else: st.error("🔴 СТРАТЕГИЯ УБЫТОЧНА ИЛИ НЕСТАБИЛЬНА")
    
    # ── Список сделок ──
    if not results['trades_df'].empty:
        with st.expander(f"📋 Журнал операций ({len(results['trades_df'])} записей — BUY и SELL отдельно)"):
            st.dataframe(results['trades_df'].tail(30), use_container_width=True)
    
    # ── PDF ──
    st.markdown("---")
    st.subheader("📄 Отчёт")
    
    if st.button("📥 Сгенерировать PDF-отчёт", use_container_width=True):
        with st.spinner("Генерация PDF..."):
            from src.reports.pdf_generator import PDFReportGenerator
            gen = PDFReportGenerator()
            filepath = gen.generate(
                metrics=m, equity_curve=results['equity_curve'], trades_df=results['trades_df'],
                signals=results['signals'], data=data,
                strategy_name=strategy_display.get(st.session_state.strategy_key, 'Strategy'),
                symbol=st.session_state.symbol, interval=st.session_state.interval,
                params=st.session_state.strategy.params if st.session_state.opt_enabled else None,
            )
            with open(filepath, "rb") as f: st.session_state.pdf_data = f.read()
            st.session_state.pdf_filename = Path(filepath).name
        st.success("PDF готов!")
    
    if st.session_state.pdf_data is not None:
        st.download_button("💾 Скачать PDF", st.session_state.pdf_data, st.session_state.pdf_filename, mime="application/pdf", use_container_width=True)
    
    # ── AI-анализ ──
    st.markdown("---")
    st.subheader("🤖 AI-анализ стратегии")
    
    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        ai_provider = st.selectbox("Провайдер", ["routerai.ru", "openai.com", "custom"])
        if ai_provider == "custom": base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
        elif ai_provider == "routerai.ru": base_url = "https://routerai.ru/api/v1"
        else: base_url = "https://api.openai.com/v1"
    with col_ai2:
        model = st.text_input("Модель", value="gpt-4o-mini" if ai_provider != "routerai.ru" else "deepseek/deepseek-v4-pro")
    
    api_key = st.text_input("API ключ", type="password", placeholder="Оставьте пустым для офлайн-анализа")
    if api_key: st.session_state.api_key = api_key
    
    if st.button("🤖 Запустить AI-анализ", use_container_width=True):
        with st.spinner("AI анализирует стратегию..."):
            from src.ai_analyst import AIAnalyst
            has_key = bool(st.session_state.get('api_key', '') or api_key)
            analyst = AIAnalyst(api_key=st.session_state.get('api_key', api_key) if has_key else "no-key", base_url=base_url, model=model)
            analysis = analyst.analyze(metrics=m, strategy_name=strategy_display.get(st.session_state.strategy_key, 'Strategy'),
                                       symbol=st.session_state.symbol, interval=st.session_state.interval)
            st.markdown("### 📝 Результат анализа")
            st.markdown(analysis)
            if not has_key: st.info("💡 Это офлайн-анализ. Подключите API ключ для AI-разбора.")
            elif "офлайн-оценка" in analysis or "автоматическая оценка" in analysis: st.warning("⚠️ AI-запрос не удался. Проверьте ключ, модель и Base URL.")
            else: st.caption(f"🤖 Модель: {model}")


st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 0.85em;'>Crypto Backtest Lab v1.0 | Оптимизатор Optuna | 10 стратегий | Данные Binance с 2017</p>", unsafe_allow_html=True)
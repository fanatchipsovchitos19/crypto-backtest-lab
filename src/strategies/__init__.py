from src.strategies.base import BaseStrategy, Signal
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.ema_ribbon import EMARibbonStrategy
from src.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategies.bollinger_bands import BollingerBandsStrategy
from src.strategies.macd_classic import MACDStrategy
from src.strategies.supertrend import SupertrendStrategy
from src.strategies.ichimoku import IchimokuStrategy
from src.strategies.donchian import DonchianStrategy
from src.strategies.psar_adx import PSARADXStrategy
from src.strategies.xgboost_ml import XGBoostStrategy

AVAILABLE_STRATEGIES = {
    'sma_crossover': SMACrossoverStrategy,
    'ema_ribbon': EMARibbonStrategy,
    'rsi_mean_reversion': RSIMeanReversionStrategy,
    'bollinger_bands': BollingerBandsStrategy,
    'macd_classic': MACDStrategy,
    'supertrend': SupertrendStrategy,
    'ichimoku': IchimokuStrategy,
    'donchian': DonchianStrategy,
    'psar_adx': PSARADXStrategy,
    'xgboost_ml': XGBoostStrategy,
}
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
import pandas as pd

@dataclass
class Signal:
    timestamp: pd.Timestamp
    signal: int
    price: float
    reason: str = ""

class BaseStrategy(ABC):
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or self.get_default_params()
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]: pass
    
    @abstractmethod
    def get_param_space(self) -> Dict[str, Any]: pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series: pass
    
    def set_params(self, params: Dict[str, Any]):
        self.params.update(params)
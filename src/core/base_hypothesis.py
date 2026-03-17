from abc import ABC, abstractmethod
import pandas as pd

class BaseHypothesis(ABC):
    def __init__(self, name: str, config: dict = None):
        self.name = name
        self.config = config or {}  # Stores the full JSON dictionary
        self.triggers = []
        self.daily_logs = []

    @abstractmethod
    def evaluate_row(self, row: pd.Series, index: pd.Timestamp):
        """
        The main router. Called by the engine for every new candle.
        Every child hypothesis MUST implement this method.
        """
        pass
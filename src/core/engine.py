import sys
from pathlib import Path
# 1. FIX: Force Python to recognize the project root (Solves ModuleNotFoundError)
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import inspect
import traceback
import pandas as pd

# 2. Dynamic Imports for the Registry
from src.library import features as core_features
from src.library import htf_features
from src.utils.macro_registry import load_macro_events

class LabEngine:
    def __init__(self, data_file: str, start_date: str, end_date: str, timeframe: str = "1h"):
        self.data_file = data_file
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.df = None
        # 3. FIX: Build the registry automatically using the @provides decorators
        self.feature_registry = self._build_dynamic_registry()

    def _build_dynamic_registry(self) -> dict:
        registry = {}
        for module in [core_features, htf_features]:
            for name, func in inspect.getmembers(module, inspect.isfunction):
                if hasattr(func, '_provides_features'):
                    for feature_name in func._provides_features:
                        registry[feature_name] = func
        return registry

    def prepare_data(self, required_features: list = None) -> bool:
        try:
            self._load_and_filter_data()
            self._apply_feature_pipeline(required_features or [])
            return True
        except Exception as e:
            print(f"❌ Critical Error in Data Pipeline: {e}")
            traceback.print_exc()
            return False

    def _load_and_filter_data(self):
        # ... (Same loading logic as before, ensure it's clean)
        self.df = pd.read_parquet(self.data_file) if self.data_file.endswith('.parquet') else pd.read_csv(self.data_file)
        if 'Datetime' in self.df.columns:
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
            self.df.set_index('Datetime', inplace=True)
        
        self.df = self.df.loc[self.start_date:self.end_date].copy()
        if self.df.index.tz is None: self.df.index = self.df.index.tz_localize('UTC')
        self.df['UA_Hour'] = self.df.index.tz_convert('Europe/Kyiv').hour

    def _apply_feature_pipeline(self, required_features: list):
        # 1. Base Core Features (ALWAYS FIRST)
        self.df = core_features.add_log_returns(self.df)
        self.df = core_features.add_atr(self.df, lookback=14)
        
        # 2. Extract unique functions and SORT THEM
        # Sorting ensures that calculate_... (C) runs before specific_... (S)
        unique_funcs = {self.feature_registry[f] for f in required_features if f in self.feature_registry}
        sorted_funcs = sorted(list(unique_funcs), key=lambda x: x.__name__)
        
        events = None
        for func in sorted_funcs:
            if 'events' in inspect.signature(func).parameters:
                if events is None: events = load_macro_events()
                self.df = func(self.df, events=events)
            else:
                self.df = func(self.df)

    def run_hypothesis(self, hypothesis):
        """The 'Dumb' Loop: Approval-aligned (No SL/TP logic here)."""
        hypothesis.reset()
        for index, row in self.df.iterrows():
            hypothesis.evaluate_row(row, index)
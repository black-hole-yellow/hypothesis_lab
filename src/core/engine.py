import os
import traceback
import pandas as pd

from src.core.evaluator import SignalEvaluator
from src.library.features import (
    add_log_returns, add_atr, add_normalized_slope, add_price_zscore, 
    add_shannon_entropy, add_hurst_exponent, add_hmm_volatility_regime, 
    add_volatility_ratio, add_volume_zscore, add_williams_fractals, 
    add_volume_profile_features, add_volatility_zscore, add_confirmed_fractals
)
# ВНИМАНИЕ: Перенесите эти функции из старого htf_features.py в новый structure_features.py
from src.library.structure_features import (
    add_previous_boundaries, calculate_multi_tf_fvgs, 
    add_1d_swing_context, add_1w_swing_context, add_weekly_swing_context,
    add_htf_trend_probability
)
from src.utils.macro_registry import load_macro_events
from src.library.feature_registry import FEATURE_REGISTRY

class LabEngine:
    def __init__(self, data_file: str, start_date: str, end_date: str, timeframe: str = "1h"):
        self.data_file = data_file
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.df = None
        self.events = load_macro_events()

    def prepare_data(self) -> bool:
        """Загружает данные и считает только БАЗОВУЮ математику."""
        try:
            self._load_and_filter_data()
            self._apply_base_pipeline()
            return True
        except Exception as e:
            print(f"❌ Critical Error in Data Pipeline: {e}")
            traceback.print_exc()
            return False

    def _load_and_filter_data(self):
        if self.data_file.endswith('.parquet'):
            temp_df = pd.read_parquet(self.data_file)
        else:
            temp_df = pd.read_csv(self.data_file)
            temp_df['Datetime'] = pd.to_datetime(temp_df['Datetime'])
            temp_df.set_index('Datetime', inplace=True)

        start, end = pd.to_datetime(self.start_date), pd.to_datetime(self.end_date)
        if temp_df.index.tz is not None:
            start, end = start.tz_localize(temp_df.index.tz), end.tz_localize(temp_df.index.tz)

        self.df = temp_df.loc[start:end].copy()
        if self.df.index.tz is None: self.df.index = self.df.index.tz_localize('UTC')
        self.df['UA_Hour'] = self.df.index.tz_convert('Europe/Kyiv').hour
        
        if self.df['Volume'].sum() == 0:
            self.df['Volume'] = abs(self.df['High'] - self.df['Low']) * 100000

    def _apply_base_pipeline(self):
        """Только ядро (Индикаторы, Фракталы, Базовые Свинги)"""
        self.df = add_log_returns(self.df)
        self.df = add_atr(self.df, lookback=14)
        self.df = add_volume_zscore(self.df, lookback=20)
        self.df = add_volatility_zscore(self.df, lookback=20)
        self.df = add_volatility_ratio(self.df, short_lookback=14, long_lookback=100)
        self.df = add_price_zscore(self.df, lookback=50)
        self.df = add_normalized_slope(self.df, lookback=20, atr_lookback=14)
        self.df = add_shannon_entropy(self.df, lookback=50)
        self.df = add_hurst_exponent(self.df, lookback=100)
        self.df = add_hmm_volatility_regime(self.df)
        self.df = add_volume_profile_features(self.df)
        self.df = calculate_multi_tf_fvgs(self.df)
        self.df = add_previous_boundaries(self.df)
        self.df = add_1w_swing_context(self.df)
        self.df = add_1d_swing_context(self.df)
        self.df = add_weekly_swing_context(self.df)
        self.df = add_williams_fractals(self.df, timeframe=self.timeframe, n=2)
        self.df = add_confirmed_fractals(self.df, n=2)
        self.df = add_htf_trend_probability(self.df, htf='4h', lookback=60)

    def apply_custom_features(self, required_features: list):
        """ДИНАМИЧЕСКАЯ ИНЪЕКЦИЯ ФИЧЕЙ ИЗ JSON"""
        applied_functions = set()
        for feature in required_features:
            if feature in FEATURE_REGISTRY:
                func = FEATURE_REGISTRY[feature]
                if func not in applied_functions:
                    self.df = func(self.df, self.events)
                    applied_functions.add(func)

    def run_hypothesis(self, hypothesis):
        """Universal Trade Manager Powered by Standardized JSON Execution Rules"""
        current_day = None
        active_trades, trades_opened_today, losses_today = [], 0, 0

        # --- ИЗВЛЕКАЕМ УНИВЕРСАЛЬНЫЕ ПРАВИЛА ИЗ JSON ---
        exec_rules = getattr(hypothesis, 'execution_rules', {})
        
        mode = exec_rules.get('mode', 'risk_reward')
        target_rr = exec_rules.get('risk_reward_ratio', 2.0)
        sl_atr_mult = exec_rules.get('sl_atr_multiplier', 1.0)
        max_hold_bars = exec_rules.get('max_hold_bars', 999)
        max_trades = exec_rules.get('max_trades_per_day', 999)
        allow_resweep = exec_rules.get('allow_resweep', False)

        for index, row in self.df.iterrows():
            day_date = index.date()
            if day_date != current_day:
                current_day, trades_opened_today, losses_today = day_date, 0, 0
                
            # ==========================================
            # 1. TRADE MANAGEMENT (Универсальное закрытие)
            # ==========================================
            still_active = []
            for trade in active_trades:
                if trade.get('Status') != 'Active': continue
                
                direction = str(trade.get('Direction', '')).capitalize()
                
                if mode == 'time_based':
                    # --- РЕЖИМ 1: Выход по Времени ---
                    trade['Hold_Bars'] = trade.get('Hold_Bars', 0) + 1
                    
                    if trade['Hold_Bars'] >= max_hold_bars:
                        close_price = row['Close']
                        if direction == 'Long':
                            trade['Outcome'] = 'Win' if close_price > trade['Entry_Price'] else 'Loss'
                        elif direction == 'Short':
                            trade['Outcome'] = 'Win' if close_price < trade['Entry_Price'] else 'Loss'
                        
                        trade['Status'], trade['Exit_Time'] = 'Closed', index
                    else:
                        still_active.append(trade)
                        
                elif mode == 'risk_reward':
                    # --- РЕЖИМ 2: Выход по Уровням (SL/TP) ---
                    high, low = row['High'], row['Low']
                    
                    if direction == 'Long':
                        if low <= trade['SL_Price']:
                            trade['Outcome'], trade['Status'], trade['Exit_Time'] = 'Loss', 'Closed', index
                            if trade.get('Datetime', index).date() == current_day: losses_today += 1
                        elif high >= trade['TP_Price']:
                            trade['Outcome'], trade['Status'], trade['Exit_Time'] = 'Win', 'Closed', index
                        else: still_active.append(trade)
                    elif direction == 'Short':
                        if high >= trade['SL_Price']:
                            trade['Outcome'], trade['Status'], trade['Exit_Time'] = 'Loss', 'Closed', index
                            if trade.get('Datetime', index).date() == current_day: losses_today += 1
                        elif low <= trade['TP_Price']:
                            trade['Outcome'], trade['Status'], trade['Exit_Time'] = 'Win', 'Closed', index
                        else: still_active.append(trade)

            active_trades = still_active

            # ==========================================
            # 2. EVALUATE SIGNAL (JSON Logic + Trend Filters)
            # ==========================================
            triggers_before = len(hypothesis.triggers)
            hypothesis.evaluate_row(row, index)

            # ==========================================
            # 3. GLOBAL GUARDS & INITIALIZATION
            # ==========================================
            if len(hypothesis.triggers) > triggers_before:
                new_trade = hypothesis.triggers[-1]
                direction = str(new_trade.get('Direction', '')).capitalize()
                
                # --- ЗАЩИТА ОТ ПЕРЕТОРГОВКИ (Circuit Breaker) ---
                limit_allows = False
                if trades_opened_today < max_trades:
                    limit_allows = True
                elif allow_resweep and trades_opened_today == max_trades and losses_today == max_trades:
                    limit_allows = True

                if not limit_allows:
                    hypothesis.triggers.pop()
                    if hasattr(hypothesis, 'daily_logs') and len(hypothesis.daily_logs) > 0: 
                        hypothesis.daily_logs.pop()
                    continue

                trades_opened_today += 1
                entry_price = row['Close']
                
                if mode == 'time_based':
                    # --- ИНИЦИАЛИЗАЦИЯ: Режим удержания по времени ---
                    new_trade.update({
                        'Entry_Price': entry_price,
                        'SL_Price': 0, 
                        'TP_Price': 0, 
                        'Status': 'Active', 
                        'Outcome': 'Pending',
                        'Hold_Bars': 0
                    })
                
                elif mode == 'risk_reward':
                    # --- ИНИЦИАЛИЗАЦИЯ: Режим снайперских уровней ---
                    sl_price = None
                    for col in self.df.columns:
                        if col.endswith('_SL') and pd.notna(row.get(col)) and row.get(col.replace('_SL',''), 0) == 1:
                            sl_price = row[col]
                            break
                    
                    if sl_price is None or pd.isna(sl_price):
                        atr_dist = row.get('ATR_14D', 0.0020) * sl_atr_mult
                        sl_price = (entry_price - atr_dist) if direction == 'Long' else (entry_price + atr_dist)
                    
                    risk = max(abs(entry_price - sl_price), row.get('ATR_14D', 0.0020))
                    tp_price = (entry_price + (target_rr * risk)) if direction == 'Long' else (entry_price - (target_rr * risk))
                    
                    new_trade.update({
                        'Entry_Price': entry_price, 
                        'SL_Price': sl_price, 
                        'TP_Price': tp_price, 
                        'Status': 'Active', 
                        'Outcome': 'Pending'
                    })
                
                active_trades.append(new_trade)
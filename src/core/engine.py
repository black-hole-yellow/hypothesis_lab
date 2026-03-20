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
from src.library.htf_features import (
    add_1d_swing_context, add_1w_level_rejection_context, add_1w_swing_context, add_asia_fvg_protection_context, add_asian_sr_alignment_context, add_boe_hawkish_context, add_election_volatility_context, 
    add_fvg_sr_confluence_context, add_geopolitical_shock_context, add_htf_trend_probability, 
    add_fvg_order_flow_context, add_london_counter_fractal_context, add_london_pdh_pdl_sweep_context, add_ny_sr_touch_context, add_previous_boundaries, add_uk_political_shock_context, add_weekly_floor_context, calculate_multi_tf_fvgs, 
    add_asian_sweep_context, add_ny_expansion_context, 
    add_weekly_swing_context
)
from src.utils.macro_registry import load_macro_events

class LabEngine:
    def __init__(self, data_file: str, start_date: str, end_date: str, timeframe: str = "1h"):
        self.data_file = data_file
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.df = None

    def prepare_data(self) -> bool:
        """Main orchestrator for loading data and building features."""
        try:
            self._load_and_filter_data()
            self._apply_feature_pipeline()

            return True
        except Exception as e:
            print(f"❌ Critical Error in Data Pipeline: {e}")
            traceback.print_exc()  # Prints the exact line where the error occurred
            return False

    def _load_and_filter_data(self):
        """Handles I/O, timezone localization, and basic data cleanup."""
        if self.data_file.endswith('.parquet'):
            temp_df = pd.read_parquet(self.data_file)
        else:
            temp_df = pd.read_csv(self.data_file)
            temp_df['Datetime'] = pd.to_datetime(temp_df['Datetime'])
            temp_df.set_index('Datetime', inplace=True)

        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        
        if temp_df.index.tz is not None:
            start = start.tz_localize(temp_df.index.tz)
            end = end.tz_localize(temp_df.index.tz)

        self.df = temp_df.loc[start:end].copy()

        if self.df.index.tz is None:
            self.df.index = self.df.index.tz_localize('UTC')
            
        # KEEP the master index in UTC for the Evaluator, 
        # but calculate the Session Hours dynamically in Kyiv Time!
        self.df['UA_Hour'] = self.df.index.tz_convert('Europe/Kyiv').hour
        

        if self.df['Volume'].sum() == 0:
            print("     ! Warning: Zero volume detected. Simulating Tick Volume...")
            self.df['Volume'] = abs(self.df['High'] - self.df['Low']) * 100000

    def _apply_feature_pipeline(self):
        """Executes the strictly ordered mathematical and structural pipeline."""
        # --- PHASE 1: FUNDAMENTAL MATH & VOLATILITY ---
        self.df = add_log_returns(self.df)
        self.df = add_atr(self.df, lookback=14)
        self.df = add_volume_zscore(self.df, lookback=20)
        self.df = add_volatility_zscore(self.df, lookback=20)
        self.df = add_volatility_ratio(self.df, short_lookback=14, long_lookback=100)
        self.df = add_price_zscore(self.df, lookback=50)

        # --- PHASE 2: ADVANCED STATISTICAL STATES ---
        self.df = add_normalized_slope(self.df, lookback=20, atr_lookback=14)
        self.df = add_shannon_entropy(self.df, lookback=50)
        self.df = add_hurst_exponent(self.df, lookback=100)
        self.df = add_hmm_volatility_regime(self.df)
        self.df = add_volume_profile_features(self.df)

        # --- PHASE 3: BASE STRUCTURE ---
        self.df = calculate_multi_tf_fvgs(self.df)
        self.df = add_previous_boundaries(self.df)
        self.df = add_1w_swing_context(self.df)
        self.df = add_1d_swing_context(self.df)
        self.df = add_weekly_swing_context(self.df)
        self.df = add_williams_fractals(self.df, timeframe=self.timeframe, n=2)
        self.df = add_confirmed_fractals(self.df, n=2)
        self.df = add_htf_trend_probability(self.df, htf='4h', lookback=60) 

        # --- PHASE 4: CONTEXTUAL & SESSION-BASED ---
        self.df = add_fvg_order_flow_context(self.df)
        self.df = add_asian_sweep_context(self.df, max_dist_pips=15)
        self.df = add_ny_sr_touch_context(self.df)

        # --- PHASE 5: MULTI-FACTOR CONFLUENCE ---
        self.df = add_ny_expansion_context(self.df)
        self.df = add_asian_sr_alignment_context(self.df, max_dist_pips=15)
        self.df = add_fvg_sr_confluence_context(self.df, max_dist_pips=150)
        self.df = add_weekly_floor_context(self.df)
        self.df = add_london_counter_fractal_context(self.df)
        self.df = add_london_pdh_pdl_sweep_context(self.df)
        self.df = add_asia_fvg_protection_context(self.df)
        self.df = add_1w_level_rejection_context(self.df, max_dist_pips=0)

        # --- PHASE 6: MACRO-EVENT CONTEXT ---
        events = load_macro_events()
        self.df = add_geopolitical_shock_context(self.df, events)
        self.df = add_election_volatility_context(self.df, events)
        self.df = add_uk_political_shock_context(self.df, events)
        self.df = add_boe_hawkish_context(self.df, events)

    def run_hypothesis(self, hypothesis):
        """Simulates the environment row-by-row for the strategy."""
        current_day = None

        for index, row in self.df.iterrows():
            day_date = index.date()
            if day_date != current_day:
                current_day = day_date
                
            # 1. Track the number of trades BEFORE evaluating the row
            triggers_before = len(hypothesis.triggers)
            
            # 2. Evaluate the row (Let the JSON check its entry rules)
            hypothesis.evaluate_row(row, index)

            # 3. --- GLOBAL TREND GUARD ---
            # If a new trade was just triggered, we inspect it!
            if len(hypothesis.triggers) > triggers_before:
                new_trade = hypothesis.triggers[-1]
                trend_prob = row.get('HTF_Bullish_Prob', 50.0)
                direction = new_trade.get('Direction', '') 
                
                is_shock = (
                    row.get('Geo_Shock_Short', 0) == 1 or 
                    row.get('Election_Vol_Crush_Short', 0) == 1 or
                    row.get('BoE_Hawkish_Long', 0) == 1
                )
                
                # The Rules of the Guard:
                if not is_shock:
                    kill_long = (direction == 'Long') and (trend_prob < 55)
                    kill_short = (direction == 'Short') and (trend_prob > 45)
                    
                    if kill_long or kill_short:
                        hypothesis.triggers.pop()
                        hypothesis.daily_logs.pop()
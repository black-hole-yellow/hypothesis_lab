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
    add_1d_swing_context, add_1w_level_rejection_context, add_1w_swing_context, add_asia_fvg_protection_context, add_asian_box_breakout_context, add_asian_sr_alignment_context, add_boe_hawkish_context, add_boe_tone_shift_proxy_context, add_cb_divergence_state_context, add_cpi_match_mean_reversion_context, add_election_volatility_context, add_fomc_sell_the_news_context, add_friday_reversal_context, 
    add_fvg_sr_confluence_context, add_geopolitical_shock_context, add_htf_trend_probability, 
    add_fvg_order_flow_context, add_judas_swing_context, add_london_counter_fractal_context, add_london_fix_fade_context, add_london_pdh_pdl_sweep_context, add_london_true_trend_context, add_macro_shock_inside_bar_context, add_monday_gap_reversion_context, add_nfp_divergence_context, add_nfp_revision_trap_context, add_ny_sr_touch_context, add_previous_boundaries, add_pure_algo_vol_crush_context, add_retail_sales_divergence_context, add_sovereign_risk_proxy_context, add_thursday_expansion_context, add_tokyo_trap_context, add_turnaround_tuesday_context, add_uk_cpi_momentum_context, add_uk_political_shock_context, add_uk_us_cpi_divergence_context, add_unemp_fakeout_context, add_wednesday_fakeout_context, add_weekend_gap_context, add_weekly_floor_context, calculate_multi_tf_fvgs, 
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
            traceback.print_exc()
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
        self.df = add_uk_cpi_momentum_context(self.df, events)
        self.df = add_weekend_gap_context(self.df)
        self.df = add_sovereign_risk_proxy_context(self.df, events)
        self.df = add_boe_tone_shift_proxy_context(self.df, events)
        self.df = add_macro_shock_inside_bar_context(self.df, events)
        self.df = add_pure_algo_vol_crush_context(self.df)
        self.df = add_nfp_divergence_context(self.df, events)
        self.df = add_nfp_revision_trap_context(self.df, events)
        self.df = add_cpi_match_mean_reversion_context(self.df, events)
        self.df = add_cb_divergence_state_context(self.df, events)
        self.df = add_fomc_sell_the_news_context(self.df, events)
        self.df = add_uk_us_cpi_divergence_context(self.df, events)
        self.df = add_unemp_fakeout_context(self.df, events)
        self.df = add_retail_sales_divergence_context(self.df, events)
        self.df = add_friday_reversal_context(self.df, events)
        self.df = add_monday_gap_reversion_context(self.df, events)
        self.df = add_turnaround_tuesday_context(self.df, events)
        self.df = add_wednesday_fakeout_context(self.df, events)
        self.df = add_thursday_expansion_context(self.df, events)
        self.df = add_london_fix_fade_context(self.df, events)
        self.df = add_tokyo_trap_context(self.df, events)
        self.df = add_asian_box_breakout_context(self.df, events)
        self.df = add_london_true_trend_context(self.df, events)
        self.df = add_judas_swing_context(self.df, events)

    def run_hypothesis(self, hypothesis):
        """Simulates the environment row-by-row with Path-Dependent 2RR Trade Management."""
        current_day = None
        active_trades = []

        for index, row in self.df.iterrows():
            
            # ==========================================
            # 1. TRADE MANAGEMENT (Check SL/TP for active trades)
            # ==========================================
            still_active = []
            for trade in active_trades:
                if trade.get('Status') != 'Active':
                    continue
                    
                high = row['High']
                low = row['Low']
                
                # Нормализуем регистр (long -> Long)
                direction = str(trade.get('Direction', '')).capitalize()
                
                if direction == 'Long':
                    if low <= trade['SL_Price']:
                        trade['Outcome'] = 'Loss'
                        trade['Status'] = 'Closed'
                        trade['Exit_Time'] = index
                    elif high >= trade['TP_Price']:
                        trade['Outcome'] = 'Win'
                        trade['Status'] = 'Closed'
                        trade['Exit_Time'] = index
                    else:
                        still_active.append(trade)
                        
                elif direction == 'Short':
                    if high >= trade['SL_Price']:
                        trade['Outcome'] = 'Loss'
                        trade['Status'] = 'Closed'
                        trade['Exit_Time'] = index
                    elif low <= trade['TP_Price']:
                        trade['Outcome'] = 'Win'
                        trade['Status'] = 'Closed'
                        trade['Exit_Time'] = index
                    else:
                        still_active.append(trade)
                else:
                    # Защита от потери сделок с неизвестным направлением
                    still_active.append(trade)

            active_trades = still_active

            # ==========================================
            # 2. EVALUATE NEW SIGNALS 
            # ==========================================
            day_date = index.date()
            if day_date != current_day:
                current_day = day_date
                
            triggers_before = len(hypothesis.triggers)
            hypothesis.evaluate_row(row, index)

            # ==========================================
            # 3. GLOBAL TREND GUARD & TRADE INITIALIZATION
            # ==========================================
            if len(hypothesis.triggers) > triggers_before:
                new_trade = hypothesis.triggers[-1]
                trend_prob = row.get('HTF_Bullish_Prob', 50.0)
                
                # Нормализуем регистр и здесь!
                direction = str(new_trade.get('Direction', '')).capitalize()
                
                is_shock = (
                    row.get('Geo_Shock_Short', 0) == 1 or 
                    row.get('Election_Vol_Crush_Short', 0) == 1 or
                    row.get('BoE_Hawkish_Long', 0) == 1 or
                    row.get('CPI_Momentum_Long', 0) == 1 or
                    row.get('CPI_Momentum_Short', 0) == 1 or
                    row.get('Gap_Up_Fade_Short', 0) == 1 or   
                    row.get('Gap_Down_Fade_Long', 0) == 1 or
                    row.get('Sovereign_Risk_Short', 0) == 1 or
                    row.get('BoE_Tone_Shift_Short', 0) == 1 or
                    row.get('Inside_Bar_Vol_Short', 0) == 1 or
                    row.get('Macro_Inside_Bar_Short', 0) == 1 or
                    row.get('Algo_Vol_Crush_Short', 0) == 1 or
                    row.get('NFP_Fade_Long', 0) == 1 or
                    row.get('NFP_Fade_Short', 0) == 1 or
                    row.get('NFP_Resumption_Long', 0) == 1 or
                    row.get('NFP_Resumption_Short', 0) == 1 or
                    row.get('CPI_Match_Fade_Short', 0) == 1 or
                    row.get('CPI_Match_Fade_Long', 0) == 1 or
                    row.get('CB_Divergence_Long', 0) == 1 or
                    row.get('CB_Divergence_Short', 0) == 1 or
                    row.get('FOMC_Sell_News_Long', 0) == 1 or
                    row.get('Macro_CPI_Div_Long', 0) == 1 or
                    row.get('Unemp_Fakeout_Long', 0) == 1 or
                    row.get('Retail_Div_Long', 0) == 1 or
                    row.get('Friday_Reversal_Short', 0) == 1 or
                    row.get('Friday_Reversal_Long', 0) == 1 or
                    row.get('Monday_Reversion_Short', 0) == 1 or
                    row.get('Monday_Reversion_Long', 0) == 1 or
                    row.get('Tuesday_Resumption_Long', 0) == 1 or
                    row.get('Tuesday_Resumption_Short', 0) == 1 or
                    row.get('Wed_Fakeout_Short', 0) == 1 or
                    row.get('Wed_Fakeout_Long', 0) == 1 or
                    row.get('Thursday_Trend_Long', 0) == 1 or
                    row.get('Thursday_Trend_Short', 0) == 1 or
                    row.get('Fix_Fade_Short', 0) == 1 or
                    row.get('Fix_Fade_Long', 0) == 1 or
                    row.get('Tokyo_Trap_Short', 0) == 1 or
                    row.get('Tokyo_Trap_Long', 0) == 1 or
                    row.get('Asian_Box_Long', 0) == 1 or
                    row.get('Asian_Box_Short', 0) == 1 or
                    row.get('LO_True_Trend_Long', 0) == 1 or
                    row.get('LO_True_Trend_Short', 0) == 1 or
                    row.get('Judas_Short', 0) == 1 or
                    row.get('Judas_Long', 0) == 1
                )
                
                # Защита по тренду
                if not is_shock:
                    kill_long = (direction == 'Long') and (trend_prob < 55)
                    kill_short = (direction == 'Short') and (trend_prob > 45)
                    
                    if kill_long or kill_short:
                        hypothesis.triggers.pop()
                        if hasattr(hypothesis, 'daily_logs') and len(hypothesis.daily_logs) > 0:
                            hypothesis.daily_logs.pop()
                        continue
                
                # --- ИНИЦИАЛИЗАЦИЯ 2RR СДЕЛКИ ---
                entry_price = row['Close']
                sl_price = None
                
                for col in self.df.columns:
                    if col.endswith('_SL') and pd.notna(row.get(col)):
                        base_feature = col.replace('_SL', '')
                        if row.get(base_feature, 0) == 1:
                            sl_price = row[col]
                            break
                
                if sl_price is None or pd.isna(sl_price):
                    atr = row.get('ATR_14D', 0.0020)
                    sl_price = (entry_price - atr) if direction == 'Long' else (entry_price + atr)
                
                # Расчет Risk и жесткого Take Profit (1:2)
                if direction == 'Long':
                    risk = entry_price - sl_price
                    if risk <= 0: risk = row.get('ATR_14D', 0.0020)
                    tp_price = entry_price + (2 * risk)
                else: # Short
                    risk = sl_price - entry_price
                    if risk <= 0: risk = row.get('ATR_14D', 0.0020)
                    tp_price = entry_price - (2 * risk)
                
                new_trade['Entry_Price'] = entry_price
                new_trade['SL_Price'] = sl_price
                new_trade['TP_Price'] = tp_price
                new_trade['Status'] = 'Active'
                new_trade['Outcome'] = 'Pending'
                
                active_trades.append(new_trade)
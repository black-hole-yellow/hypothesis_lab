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
    add_weekly_swing_context, add_ny_continuation_context, add_ny_news_sweep_context
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
        try:
            self._load_and_filter_data()
            self._apply_feature_pipeline()
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
            self.df['Volume'] = abs(self.df['High'] - self.df['Low']) * 100000

    def _apply_feature_pipeline(self):
        # --- PHASE 1-3 ---
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

        # --- PHASE 4-5 ---
        self.df = add_fvg_order_flow_context(self.df)
        self.df = add_asian_sweep_context(self.df, max_dist_pips=15)
        self.df = add_ny_sr_touch_context(self.df)
        self.df = add_ny_expansion_context(self.df)
        self.df = add_asian_sr_alignment_context(self.df, max_dist_pips=15)
        self.df = add_fvg_sr_confluence_context(self.df, max_dist_pips=150)
        self.df = add_weekly_floor_context(self.df)
        self.df = add_london_counter_fractal_context(self.df)
        self.df = add_london_pdh_pdl_sweep_context(self.df)
        self.df = add_asia_fvg_protection_context(self.df)
        self.df = add_1w_level_rejection_context(self.df, max_dist_pips=0)

        # --- PHASE 6 ---
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
        self.df = add_ny_continuation_context(self.df, events)
        self.df = add_ny_news_sweep_context(self.df, events)
        self.df = add_london_fix_fade_context(self.df, events)

    def run_hypothesis(self, hypothesis):
        """Path-Dependent 1:2 RR with 1-trade-per-day (+1 Resweep) Limit."""
        current_day = None
        active_trades = []
        trades_opened_today = 0
        losses_today = 0

        for index, row in self.df.iterrows():
            day_date = index.date()
            if day_date != current_day:
                current_day = day_date
                trades_opened_today = 0
                losses_today = 0
                
            # 1. TRADE MANAGEMENT (Закрытие 1:2 RR)
            still_active = []
            for trade in active_trades:
                if trade.get('Status') != 'Active': continue
                high, low = row['High'], row['Low']
                direction = str(trade.get('Direction', '')).capitalize()
                
                if direction == 'Long':
                    if low <= trade['SL_Price']:
                        trade['Outcome'], trade['Status'] = 'Loss', 'Closed'
                        trade['Exit_Time'] = index
                        if trade.get('Datetime', index).date() == current_day: losses_today += 1
                    elif high >= trade['TP_Price']:
                        trade['Outcome'], trade['Status'] = 'Win', 'Closed'
                        trade['Exit_Time'] = index
                    else: still_active.append(trade)
                elif direction == 'Short':
                    if high >= trade['SL_Price']:
                        trade['Outcome'], trade['Status'] = 'Loss', 'Closed'
                        trade['Exit_Time'] = index
                        if trade.get('Datetime', index).date() == current_day: losses_today += 1
                    elif low <= trade['TP_Price']:
                        trade['Outcome'], trade['Status'] = 'Win', 'Closed'
                        trade['Exit_Time'] = index
                    else: still_active.append(trade)
            active_trades = still_active

            # 2. EVALUATE SIGNAL
            triggers_before = len(hypothesis.triggers)
            hypothesis.evaluate_row(row, index)

            # 3. GLOBAL GUARDS & INITIALIZATION
            if len(hypothesis.triggers) > triggers_before:
                new_trade = hypothesis.triggers[-1]
                direction = str(new_trade.get('Direction', '')).capitalize()
                
                # Контр-трендовые сигналы (пропускают фильтр HTF-тренда)
                is_shock = (
                    row.get('Judas_Short', 0) == 1 or
                    row.get('Judas_Long', 0) == 1 or
                    row.get('NY_Sweep_Short', 0) == 1 or  
                    row.get('NY_Sweep_Long', 0) == 1 or
                    row.get('Fix_Fade_Short', 0) == 1 or  
                    row.get('Fix_Fade_Long', 0) == 1        
                )

                # Защита по тренду (Только для базовых стратегий)
                if not is_shock:
                    trend_prob = row.get('HTF_Bullish_Prob', 50.0)
                    kill_long = (direction == 'Long') and (trend_prob < 55)
                    kill_short = (direction == 'Short') and (trend_prob > 45)
                    
                    if kill_long or kill_short:
                        hypothesis.triggers.pop()
                        if hasattr(hypothesis, 'daily_logs') and len(hypothesis.daily_logs) > 0:
                            hypothesis.daily_logs.pop()
                        continue
                
                # Защита от переторговки (Circuit Breaker)
                limit_allows = (trades_opened_today == 0) or (trades_opened_today == 1 and losses_today == 1)

                if not limit_allows:
                    hypothesis.triggers.pop()
                    if hasattr(hypothesis, 'daily_logs') and len(hypothesis.daily_logs) > 0:
                        hypothesis.daily_logs.pop()
                    continue

                trades_opened_today += 1
                
                # ИНИЦИАЛИЗАЦИЯ 1:2 RR
                entry_price = row['Close']
                sl_price = None
                for col in self.df.columns:
                    if col.endswith('_SL') and pd.notna(row.get(col)) and row.get(col.replace('_SL',''), 0) == 1:
                        sl_price = row[col]
                        break
                
                if sl_price is None or pd.isna(sl_price):
                    atr = row.get('ATR_14D', 0.0020)
                    sl_price = (entry_price - atr) if direction == 'Long' else (entry_price + atr)
                
                risk = abs(entry_price - sl_price)
                if risk <= 0: risk = row.get('ATR_14D', 0.0020)
                
                tp_price = (entry_price + 2.0 * risk) if direction == 'Long' else (entry_price - 2.0 * risk)
                
                new_trade.update({
                    'Entry_Price': entry_price, 'SL_Price': sl_price, 'TP_Price': tp_price,
                    'Status': 'Active', 'Outcome': 'Pending'
                })
                active_trades.append(new_trade)
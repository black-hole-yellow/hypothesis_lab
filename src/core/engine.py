import os
import pandas as pd

from src.utils.data_loader import load_and_prep_data
from src.core.evaluator import SignalEvaluator, save_hypothesis_results
from src.library.features import (
    add_log_returns, add_atr, add_normalized_slope, add_price_zscore, 
    add_shannon_entropy, add_hurst_exponent, add_hmm_volatility_regime, 
    add_volatility_ratio, add_volume_zscore, add_williams_fractals, add_volume_profile_features, add_volatility_zscore, add_confirmed_fractals
)
from src.library.htf_features import (add_1w_swing_context, add_asian_sr_alignment_context, 
                                      add_fvg_sr_confluence_context, 
                                      add_htf_trend_probability, 
                                      add_fvg_order_flow_context, 
                                      add_ny_sr_touch_context, add_weekly_floor_context,calculate_multi_tf_fvgs, 
                                      add_fvg_order_flow_context, 
                                      add_asian_sweep_context, 
                                      add_asian_sweep_context,
                                      add_ny_expansion_context, 
                                      add_weekly_swing_context,
)

class LabEngine:
    def __init__(self, data_file: str, start_date: str, end_date: str, timeframe: str = "1h"):
        self.data_file = data_file
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.df = None

    def prepare_data(self):
        try:
            # 1. Load into a temporary local variable
            if self.data_file.endswith('.parquet'):
                temp_df = pd.read_parquet(self.data_file)
            else:
                temp_df = pd.read_csv(self.data_file)
                temp_df['Datetime'] = pd.to_datetime(temp_df['Datetime'])
                temp_df.set_index('Datetime', inplace=True)

            # 2. Handle Timezones securely before slicing
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            
            # If the data index has a timezone (like our Parquet), make start/end aware too!
            if temp_df.index.tz is not None:
                start = start.tz_localize(temp_df.index.tz)
                end = end.tz_localize(temp_df.index.tz)

            # 3. Filter dates securely
            temp_df = temp_df.loc[start:end].copy()

            # 4. Assign the clean, filtered data to the engine
            self.df = temp_df

            if self.df.index.tz is None:
                # If naive (CSV), assume UTC before converting to Kyiv
                self.df.index = self.df.index.tz_localize('UTC')
            
            self.df['UA_Hour'] = self.df.index.tz_convert('Europe/Kyiv').hour

            if self.df['Volume'].sum() == 0:
                print("     ! Warning: Zero volume detected. Simulating Tick Volume...")
                self.df['Volume'] = abs(self.df['High'] - self.df['Low']) * 100000

            # --- PHASE 1: FUNDAMENTAL MATH & VOLATILITY ---
            # These must come first as they provide the raw "physics" for everything else.
            self.df = add_log_returns(self.df)
            self.df = add_atr(self.df, lookback=14)
            self.df = add_volume_zscore(self.df, lookback=20)
            self.df = add_volatility_zscore(self.df, lookback=20)
            self.df = add_volatility_ratio(self.df, short_lookback=14, long_lookback=100)
            self.df = add_price_zscore(self.df, lookback=50)

            # --- PHASE 2: ADVANCED STATISTICAL STATES ---
            # Descriptors of the current market regime.
            self.df = add_normalized_slope(self.df, lookback=20, atr_lookback=14)
            self.df = add_shannon_entropy(self.df, lookback=50)
            self.df = add_hurst_exponent(self.df, lookback=100)
            self.df = add_hmm_volatility_regime(self.df)
            self.df = add_volume_profile_features(self.df)

            # --- PHASE 3: BASE STRUCTURE (The "Providers") ---
            # These generate the columns that the more complex "Strategy" features need.
            self.df = calculate_multi_tf_fvgs(self.df)         # Provides FVG zones
            self.df = add_1w_swing_context(self.df)            # Provides 1W Swing Direction
            self.df = add_weekly_swing_context(self.df)         # (Redundant, but kept if distinct)
            self.df = add_williams_fractals(self.df, timeframe=self.timeframe, n=2)
            self.df = add_confirmed_fractals(self.df, n=2)     # Provides Bull/Bear Fractals
            self.df = add_htf_trend_probability(self.df, htf='4h', lookback=60) 

            # --- PHASE 4: CONTEXTUAL & SESSION-BASED (The "Consumers") ---
            # These depend on the math and structure calculated in Phases 1-3.
            self.df = add_fvg_order_flow_context(self.df)      # Needs FVG
            self.df = add_asian_sweep_context(self.df, max_dist_pips=15) # Provides Asian High/Low
            self.df = add_ny_sr_touch_context(self.df)         # Provides Major Resistance/Support

            # --- PHASE 5: MULTI-FACTOR CONFLUENCE ---
            # The most complex logic that requires both Session Levels AND Structural States.
            self.df = add_ny_expansion_context(self.df)        # Needs Asian High/Low
            self.df = add_asian_sr_alignment_context(self.df, max_dist_pips=15) # Needs Asian + SR
            self.df = add_fvg_sr_confluence_context(self.df, max_dist_pips=150) # Needs FVG + SR

            fractal_cols = [c for c in self.df.columns if 'fractal' in c.lower() or 'bull' in c.lower()]
            print(f"AVAILABLE FRACTAL COLUMNS: {fractal_cols}")
            
            self.df = add_weekly_floor_context(self.df)        # Needs 1W Swing + Fractals
            

            self.df.dropna(inplace=True)
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

        return True

    def run_hypothesis(self, hypothesis):
        current_day = None

        for index, row in self.df.iterrows():
            day_date = index.date()
            if day_date != current_day:
                current_day = day_date

            hypothesis.evaluate_row(row, index)

        # Save dynamic audit log named after the hypothesis!
        os.makedirs("output", exist_ok=True)
        audit_filename = f"output/{hypothesis.name}_audit_log.csv"
        pd.DataFrame(hypothesis.daily_logs).to_csv(audit_filename, index=False)

    def evaluate(self, hypothesis):
        import json # Make sure json is imported at the top of engine.py
        
        if len(hypothesis.triggers) == 0:
            print("❌ No triggers generated.")
            return

        evaluator = SignalEvaluator(self.df, hypothesis.triggers, hypothesis.name)
        metrics = evaluator.calculate_metrics()
        
        for key, value in metrics.items():
            print(f"  {key:<20} : {value}")
        
        

        # --- THE PRODUCTION HANDOFF AUTOMATION ---
        # Assuming your SignalEvaluator returns metrics like 'T_Stat' and 'Win_Rate'
        t_stat = metrics.get('T_Stat', 0)
        
        # Define your strict production thresholds here
        if t_stat >= 2.0:
            print("✅ STRATEGY VALIDATED! Generating Production Artifacts...")
            
            config_filename = f"configs/production/{hypothesis.name}.json"
            
            # Save the exact configuration that passed the test
            with open(config_filename, "w") as f:
                json.dump(hypothesis.config, f, indent=4)
                
            print(f"      -> Production JSON Config saved to: {config_filename}")
        else:
            print("⚠️ Strategy failed statistical validation (T-Stat < 2.0). No production config generated.")
import pandas as pd
from src.core.base_hypothesis import BaseHypothesis

class LondonFakeMove(BaseHypothesis):
    def __init__(self, config: dict):
        # Dynamically set the name from the JSON file
        super().__init__(name=config.get("strategy_name", "London_Pro_Trend_Sweep"), config=config) 
        
        # Load parameters explicitly for clean code below
        self.p = self.config["parameters"]
        
        # State variables
        self.session_date = None
        self.asian_high = 0.0
        self.asian_low = float('inf')
        self.session_trend_prob = 50.0 
        
        # Fractal Storage
        self.asian_up_fractals = []
        self.asian_down_fractals = []
        
        # Execution State
        self.is_asian_range_locked = False
        self.lo_sweep_type = "None"
        self.lo_sweep_time = None

    def evaluate_row(self, row, index):
        hour = index.hour

        # ---------------------------------------------------------
        # 1. EVALUATE OUTCOME AT NY CLOSE & RESET
        # ---------------------------------------------------------
        if hour == self.p["eval_end_hour"]:
            if self.session_date is not None:
                day_close = row['Close']
                
                actual_direction = "Inside Range"
                if day_close < self.asian_low:
                    actual_direction = "Short"  
                elif day_close > self.asian_high:
                    actual_direction = "Long"   

                hypothesis_success = False
                if self.session_trend_prob > self.p["bullish_trend_threshold"]: 
                    if self.lo_sweep_type == "Lower_Sweep" and actual_direction == "Long":
                        hypothesis_success = True
                elif self.session_trend_prob < self.p["bullish_trend_threshold"]: 
                    if self.lo_sweep_type == "Upper_Sweep" and actual_direction == "Short":
                        hypothesis_success = True

                log_date = row['UA_Time'].date() if 'UA_Time' in row else self.session_date

                if self.lo_sweep_type != "None":
                    self.daily_logs.append({
                        'Session_Date': log_date,
                        'Trend_Bullish_Prob_%': self.session_trend_prob,
                        'Asian_High': self.asian_high,
                        'Asian_Low': self.asian_low,
                        'Sweep_Type': self.lo_sweep_type,
                        'Sweep_Time': self.lo_sweep_time,
                        'Close_at_NY_End': day_close,
                        'Day_Direction': actual_direction,
                        'Hypothesis_Success': hypothesis_success
                    })

            # --- RESET EVERYTHING FOR THE NEW SESSION ---
            self.session_date = (index + pd.Timedelta(days=1)).date()
            self.asian_high = row['High']
            self.asian_low = row['Low']
            self.asian_up_fractals = []
            self.asian_down_fractals = []
            self.is_asian_range_locked = False
            self.lo_sweep_type = "None"
            self.lo_sweep_time = None
            
            self.session_trend_prob = row.get('HTF_Bullish_Prob', 50.0)

        # ---------------------------------------------------------
        # 2. BUILD ASIAN RANGE & COLLECT FRACTALS 
        # ---------------------------------------------------------
        if (hour >= self.p["asian_start_hour"]) or (hour < self.p["asian_end_hour"]):
            if row['High'] > self.asian_high: self.asian_high = row['High']
            if row['Low'] < self.asian_low: self.asian_low = row['Low']
                
            if row.get('Fractal_Up') == True:
                self.asian_up_fractals.append(row['High'])
            if row.get('Fractal_Down') == True:
                self.asian_down_fractals.append(row['Low'])

        # ---------------------------------------------------------
        # 3. LOCK ASIAN RANGE
        # ---------------------------------------------------------
        if hour == self.p["asian_end_hour"]:
            self.is_asian_range_locked = True
            if not self.asian_up_fractals: self.asian_up_fractals.append(self.asian_high)
            if not self.asian_down_fractals: self.asian_down_fractals.append(self.asian_low)

        # ---------------------------------------------------------
        # 4. PRO-TREND SWEEP DETECTION 
        # ---------------------------------------------------------
        if self.is_asian_range_locked and self.lo_sweep_type == "None" and (self.p["eval_start_hour"] <= hour < self.p["eval_end_hour"]):
            
            swept_upper = any(row['High'] > fractal for fractal in self.asian_up_fractals)
            swept_lower = any(row['Low'] < fractal for fractal in self.asian_down_fractals)
            
            ua_time_str = row['UA_Time'].strftime('%H:%M:%S') if 'UA_Time' in row else (index + pd.Timedelta(hours=2)).strftime('%H:%M:%S')
            
            is_bullish = self.session_trend_prob > self.p["bullish_trend_threshold"]
            is_bearish = self.session_trend_prob < self.p["bullish_trend_threshold"]

            if is_bearish and swept_upper:
                self.lo_sweep_type = "Upper_Sweep"
                self.lo_sweep_time = ua_time_str
                self.triggers.append({'Datetime': index, 'Direction': 'Short'})
                
            elif is_bullish and swept_lower:
                self.lo_sweep_type = "Lower_Sweep"
                self.lo_sweep_time = ua_time_str
                self.triggers.append({'Datetime': index, 'Direction': 'Long'})
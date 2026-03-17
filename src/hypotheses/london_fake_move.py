import pandas as pd

class LondonFakeMove:
    def __init__(self, name="Pro_Trend_Sweep_Daily_Profile"):
        self.name = name
        self.triggers = []
        self.daily_logs = [] 
        
        # State variables
        self.session_date = None
        self.asian_high = 0.0
        self.asian_low = float('inf')
        self.session_trend_prob = 50.0  # <--- NEW: Stores the Trend Score
        
        # Fractal Storage
        self.asian_up_fractals = []
        self.asian_down_fractals = []
        
        # Execution State
        self.is_asian_range_locked = False
        self.lo_sweep_type = "None"
        self.lo_sweep_time = None

    def evaluate_row(self, row, index):
        # We assume index is EST based on the data_loader
        hour = index.hour

        # ---------------------------------------------------------
        # 1. EVALUATE THE OUTCOME AT NY CLOSE (16:00 EST) & RESET
        # ---------------------------------------------------------
        if hour == 16:
            if self.session_date is not None:
                day_close = row['Close']
                
                # Determine ACTUAL direction of the day based on the NY Close
                actual_direction = "Inside Range"
                if day_close < self.asian_low:
                    actual_direction = "Short"  # Closed below Asia
                elif day_close > self.asian_high:
                    actual_direction = "Long"   # Closed above Asia

                # PRO-TREND SUCCESS LOGIC: 
                # Did it sweep the correct side based on trend, and close in profit?
                hypothesis_success = False
                if self.session_trend_prob > 50.0: # Bullish Trend
                    if self.lo_sweep_type == "Lower_Sweep" and actual_direction == "Long":
                        hypothesis_success = True
                elif self.session_trend_prob < 50.0: # Bearish Trend
                    if self.lo_sweep_type == "Upper_Sweep" and actual_direction == "Short":
                        hypothesis_success = True

                # Use UA_Time column if available, else fallback to standard date
                log_date = row['UA_Time'].date() if 'UA_Time' in row else self.session_date

                # Save to Audit Log
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
            
            # Lock in the HTF Trend for the upcoming Asian/London/NY session
            self.session_trend_prob = row.get('HTF_Bullish_Prob', 50.0)

        # ---------------------------------------------------------
        # 2. BUILD ASIAN RANGE & COLLECT FRACTALS (18:00 to 02:59 EST)
        # ---------------------------------------------------------
        if (hour >= 18) or (hour < 3):
            # Track Absolute High/Low
            if row['High'] > self.asian_high: self.asian_high = row['High']
            if row['Low'] < self.asian_low: self.asian_low = row['Low']
                
            # Track Williams Fractals
            if row.get('Fractal_Up') == True:
                self.asian_up_fractals.append(row['High'])
            if row.get('Fractal_Down') == True:
                self.asian_down_fractals.append(row['Low'])

        # ---------------------------------------------------------
        # 3. LOCK ASIAN RANGE AT LONDON OPEN (03:00 EST)
        # ---------------------------------------------------------
        if hour == 3:
            self.is_asian_range_locked = True
            # Fallback: If no fractals formed, use the absolute high/low
            if not self.asian_up_fractals: self.asian_up_fractals.append(self.asian_high)
            if not self.asian_down_fractals: self.asian_down_fractals.append(self.asian_low)

        # ---------------------------------------------------------
        # 4. PRO-TREND SWEEP DETECTION (London & NY: 03:00 to 15:59 EST)
        # ---------------------------------------------------------
        if self.is_asian_range_locked and self.lo_sweep_type == "None" and (3 <= hour < 16):
            
            swept_upper = any(row['High'] > fractal for fractal in self.asian_up_fractals)
            swept_lower = any(row['Low'] < fractal for fractal in self.asian_down_fractals)
            
            # Calculate Ukraine Time for the log seamlessly
            ua_time_str = row['UA_Time'].strftime('%H:%M:%S') if 'UA_Time' in row else (index + pd.Timedelta(hours=2)).strftime('%H:%M:%S')
            
            # --- THE NEW TREND FILTER ---
            is_bullish = self.session_trend_prob > 50.0
            is_bearish = self.session_trend_prob < 50.0

            if is_bearish and swept_upper:
                # Bearish Trend -> Ignore lower sweeps -> Look for Sweep of Highs -> Go Short
                self.lo_sweep_type = "Upper_Sweep"
                self.lo_sweep_time = ua_time_str
                self.triggers.append({'Datetime': index, 'Direction': 'Short'})
                
            elif is_bullish and swept_lower:
                # Bullish Trend -> Ignore upper sweeps -> Look for Sweep of Lows -> Go Long
                self.lo_sweep_type = "Lower_Sweep"
                self.lo_sweep_time = ua_time_str
                self.triggers.append({'Datetime': index, 'Direction': 'Long'})
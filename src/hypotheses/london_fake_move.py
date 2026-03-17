import pandas as pd

class LondonFakeMove:
    def __init__(self, name="London_Sweep_Daily_Profile"):
        self.name = name
        self.triggers = []
        self.daily_logs = [] 
        
        # State variables
        self.session_date = None
        self.asian_high = 0.0
        self.asian_low = float('inf')
        
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
        # 1. EVALUATE THE OUTCOME AT 18:00 & RESET FOR NEW DAY
        # ---------------------------------------------------------
        if hour == 11:
            if self.session_date is not None:
                day_close = row['Close']
                
                # Determine ACTUAL direction of the day based on the 18:00 Close
                actual_direction = "Inside Range"
                if day_close < self.asian_low:
                    actual_direction = "Short"  # Closed below Asia
                elif day_close > self.asian_high:
                    actual_direction = "Long"   # Closed above Asia

                # Did the hypothesis play out perfectly?
                # (Swept an Upper Fractal -> Closed Short)
                hypothesis_success = False
                if self.lo_sweep_type == "Upper_Sweep" and actual_direction == "Short":
                    hypothesis_success = True
                elif self.lo_sweep_type == "Lower_Sweep" and actual_direction == "Long":
                    hypothesis_success = True

                # Save to Audit Log
                self.daily_logs.append({
                    'Session_Date': self.session_date,
                    'Asian_High': self.asian_high,
                    'Asian_Low': self.asian_low,
                    'LO_Sweep_Type': self.lo_sweep_type,
                    'LO_Sweep_Time': self.lo_sweep_time,
                    'Close_at_1800': day_close,
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

            self.ua_offset = pd.Timedelta(hours=2)

        # ---------------------------------------------------------
        # 2. BUILD ASIAN RANGE & COLLECT FRACTALS (18:00 to 01:59)
        # ---------------------------------------------------------
        if (hour >= 18) or (hour < 2):
            # Track Absolute High/Low
            if row['High'] > self.asian_high: self.asian_high = row['High']
            if row['Low'] < self.asian_low: self.asian_low = row['Low']
                
            # Track Williams Fractals
            if row.get('Fractal_Up') == True:
                self.asian_up_fractals.append(row['High'])
            if row.get('Fractal_Down') == True:
                self.asian_down_fractals.append(row['Low'])

        # ---------------------------------------------------------
        # 3. LOCK ASIAN RANGE (02:00)
        # ---------------------------------------------------------
        if hour == 2:
            self.is_asian_range_locked = True
            # Fallback: If no fractals formed, use the absolute high/low
            if not self.asian_up_fractals: self.asian_up_fractals.append(self.asian_high)
            if not self.asian_down_fractals: self.asian_down_fractals.append(self.asian_low)

        # ---------------------------------------------------------
        # 4. DETECT LONDON FRACTAL SWEEP (02:00 to 07:59)
        # ---------------------------------------------------------
        if self.is_asian_range_locked and self.lo_sweep_type == "None" and (2 <= hour < 8):
            
            swept_upper = any(row['High'] > fractal for fractal in self.asian_up_fractals)
            swept_lower = any(row['Low'] < fractal for fractal in self.asian_down_fractals)
            
            # Calculate Ukraine Time for the log
            ua_time = (index + self.ua_offset).strftime('%H:%M:%S')
            
            if swept_upper and swept_lower:
                self.lo_sweep_type = "Both_Swept"
                self.lo_sweep_time = ua_time
            elif swept_upper:
                self.lo_sweep_type = "Upper_Sweep"
                self.lo_sweep_time = ua_time
                self.triggers.append({'Datetime': index, 'Direction': 'Short'})
            elif swept_lower:
                self.lo_sweep_type = "Lower_Sweep"
                self.lo_sweep_time = ua_time
                self.triggers.append({'Datetime': index, 'Direction': 'Long'})
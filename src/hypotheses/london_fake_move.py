import pandas as pd

class LondonFakeMove:
    def __init__(self, name="London_Initial_Sweep_Raw"):
        self.name = name
        self.triggers = []
        self.daily_logs = []  # <-- NEW: Stores the daily audit trail
        
        # State variables
        self.asian_high = 0.0
        self.asian_low = float('inf')
        self.is_asian_range_locked = False
        self.sweep_fired_today = False
        
        # Tracking variables for the Audit Log
        self.session_date = None
        self.trigger_time = "No Trigger"
        self.trigger_direction = "None"

    def evaluate_row(self, row, index):
        hour = index.hour

        # 1. Reset at the START of the Asian Session (18:00)
        if hour == 18:
            # --- SAVE PREVIOUS DAY TO AUDIT LOG ---
            if self.session_date is not None:
                self.daily_logs.append({
                    'Session_Date': self.session_date,
                    'Asian_High': self.asian_high,
                    'Asian_Low': self.asian_low,
                    'Signal_Fired': self.sweep_fired_today,
                    'Trigger_Time': self.trigger_time,
                    'Direction': self.trigger_direction
                })

            # --- RESET FOR THE NEW DAY ---
            # If it's 18:00 on Monday, the actual London trading session is on Tuesday
            self.session_date = (index + pd.Timedelta(days=1)).date()
            self.asian_high = row['High']
            self.asian_low = row['Low']
            self.is_asian_range_locked = False
            self.sweep_fired_today = False
            self.trigger_time = "No Trigger"
            self.trigger_direction = "None"

        # 2. Build the Asian Range (18:00 to 01:59)
        if (hour >= 18) or (hour < 2):
            if row['High'] > self.asian_high:
                self.asian_high = row['High']
            if row['Low'] < self.asian_low:
                self.asian_low = row['Low']

        # 3. Lock the range at London Open (02:00)
        if hour == 2:
            self.is_asian_range_locked = True

        # 4. Detect the INITIAL London Open Sweep (02:00 to 07:59)
        if self.is_asian_range_locked and not self.sweep_fired_today and (2 <= hour < 8):
            
            # Sweeps Asian High -> Reverses DOWN
            if row['High'] > self.asian_high:
                self.triggers.append({'Datetime': index, 'Direction': 'Short'})
                self.sweep_fired_today = True
                self.trigger_time = index.time()
                self.trigger_direction = "Short"

            # Sweeps Asian Low -> Reverses UP
            elif row['Low'] < self.asian_low:
                self.triggers.append({'Datetime': index, 'Direction': 'Long'})
                self.sweep_fired_today = True
                self.trigger_time = index.time()
                self.trigger_direction = "Long"
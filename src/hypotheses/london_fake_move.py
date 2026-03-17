import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.base_hypothesis import BaseHypothesis, State

class LondonFakeMove:
    def __init__(self, name="London_Fake_Move_V1"):
        self.name = name
        self.triggers = []
        
        # State variables
        self.asian_high = 0.0
        self.asian_low = float('inf')
        self.is_asian_range_locked = False

    def evaluate_row(self, row, index):
        hour = index.hour

        # 1. Reset at the START of the Asian Session (18:00)
        if hour == 18:
            self.asian_high = row['High']
            self.asian_low = row['Low']
            self.is_asian_range_locked = False

        # 2. Build the Asian Range (18:00 to 02:59)
        if (hour >= 18) or (hour < 3):
            if row['High'] > self.asian_high:
                self.asian_high = row['High']
            if row['Low'] < self.asian_low:
                self.asian_low = row['Low']

        # 3. Lock the range right as London Opens
        # London overlaps with Asia at hour == 2, so the range is essentially formed
        if hour == 2:
            self.is_asian_range_locked = True

        # 4. Detect the London Open Sweep (02:00 to 07:59)
        if self.is_asian_range_locked and (hour >= 2 and hour < 8):
            
            # --- QUANTITATIVE FILTERS ---
            # Is the market actually trending? (Hurst > 0.5)
            # Is the market volatile enough to trade? (Vol_Ratio > 1.0)
            is_trending = row.get('Hurst', 0) > 0.5 
            is_awake = row.get('Vol_Ratio', 0) > 1.0
            
            if not (is_trending and is_awake):
                return # Skip if the market is chopping or dead
                
            # Define Trend Direction 
            trend_is_up = row.get('HTF_Trend_Up', False)
            trend_is_down = row.get('HTF_Trend_Down', False)

            # --- HYPOTHESIS EXECUTION ---
            
            # BULLISH TRIGGER: Trend is UP, Price drops to sweep Asian LOW
            if trend_is_up and row['Low'] < self.asian_low:
                self.triggers.append({
                    'Datetime': index,
                    'Direction': 'Long'
                })
                # Lock out further triggers for this specific session
                self.is_asian_range_locked = False 

            # BEARISH TRIGGER: Trend is DOWN, Price rallies to sweep Asian HIGH
            elif trend_is_down and row['High'] > self.asian_high:
                self.triggers.append({
                    'Datetime': index,
                    'Direction': 'Short'
                })
                # Lock out further triggers for this specific session
                self.is_asian_range_locked = False
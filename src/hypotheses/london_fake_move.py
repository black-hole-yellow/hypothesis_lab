import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.base_hypothesis import BaseHypothesis, State

class LondonFakeMove(BaseHypothesis):
    def __init__(self):
        super().__init__(
            hypothesis_id="LDN_FAKE_001", 
            description="London sweeps Asia, closes past origin fractal, targets opposite Asia boundary", 
            variables=["Asia_High", "Asia_Low", "Origin_Fractal", "Session"]
        )
        
        # Internal Tracking Variables
        self.last_fractal_high = None
        self.last_fractal_low = None
        
        self.asia_high = -np.inf
        self.asia_low = np.inf
        self.origin_for_high = None
        self.origin_for_low = None
        
        self.swept_side = None      # 'HIGH' or 'LOW'
        self.sweep_extreme = None   # The highest/lowest point of the London sweep (Invalidation level)
        self.current_session = None

    def _reset_daily_trackers(self):
        """Clears the session variables at the start of a new Asia session."""
        self.asia_high = -np.inf
        self.asia_low = np.inf
        self.origin_for_high = None
        self.origin_for_low = None
        self.swept_side = None
        self.sweep_extreme = None

    def check_trigger(self, index: pd.Timestamp, row: pd.Series):
        """
        Phase 1 & Phase 2: Observes Asia to build the structural levels. 
        Watches London for the Sweep and the Body Close Confirmation.
        """
        # Update rolling fractals
        if row.get('Fractal_High'):
            self.last_fractal_high = row['High']
        if row.get('Fractal_Low'):
            self.last_fractal_low = row['Low']

        # Detect Session Transition
        if row['Session'] != self.current_session:
            if row['Session'] == 'Asia':
                self._reset_daily_trackers()
                # If a previous hypothesis is still pending from yesterday, reset the base state
                if self.state == State.PENDING:
                    self.reset()
            self.current_session = row['Session']

        # PHASE 1: Build Asia Levels
        if self.current_session == 'Asia':
            if row['High'] > self.asia_high:
                self.asia_high = row['High']
                self.origin_for_high = self.last_fractal_low # The red line in your diagram
            
            if row['Low'] < self.asia_low:
                self.asia_low = row['Low']
                self.origin_for_low = self.last_fractal_high

        # PHASE 2 & 3: London Sweep & Confirmation
        elif self.current_session == 'London':
            # Ensure we actually have valid Asia levels to sweep
            if self.asia_high == -np.inf or self.origin_for_high is None:
                return

            # Check for the initial sweep
            if self.swept_side is None:
                if row['High'] > self.asia_high:
                    self.swept_side = 'HIGH'
                    self.sweep_extreme = row['High']
                elif row['Low'] < self.asia_low:
                    self.swept_side = 'LOW'
                    self.sweep_extreme = row['Low']

            # Check for Fake Move Confirmation (Body Close)
            if self.swept_side == 'HIGH':
                self.sweep_extreme = max(self.sweep_extreme, row['High'])
                if row['Close'] < self.origin_for_high:
                    self.mark_triggered(index, row) # Fake move confirmed!
                    
            elif self.swept_side == 'LOW':
                self.sweep_extreme = min(self.sweep_extreme, row['Low'])
                if row['Close'] > self.origin_for_low:
                    self.mark_triggered(index, row) # Fake move confirmed!

    def update_state(self, index: pd.Timestamp, row: pd.Series):
        """
        Phase 4: Tracks the "Real Move". 
        Target: Opposite Asia boundary.
        Invalidation: Price re-breaks the London sweep extreme.
        Timeout: New Asia session starts.
        """
        if self.swept_side == 'HIGH':
            # Target Hit
            if row['Low'] <= self.asia_low:
                self.mark_completed(True)
            # Stop Loss / Invalidation Hit
            elif row['High'] > self.sweep_extreme:
                self.mark_completed(False)
                
        elif self.swept_side == 'LOW':
            # Target Hit
            if row['High'] >= self.asia_high:
                self.mark_completed(True)
            # Stop Loss / Invalidation Hit
            elif row['Low'] < self.sweep_extreme:
                self.mark_completed(False)

        # Timeout constraint
        if row['Session'] == 'Asia':
            self.mark_completed(False)

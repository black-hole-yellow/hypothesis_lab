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
        Phase 1-3: Builds Asia, watches London for sweep, and confirms 
        the fake move in London or New York.
        """
        # DNA: Update rolling fractals
        if row.get('Fractal_High'):
            self.last_fractal_high = row['High']
        if row.get('Fractal_Low'):
            self.last_fractal_low = row['Low']

        # Session Transition Logic
        if row['Session'] != self.current_session:
            if row['Session'] == 'Asia':
                self._reset_daily_trackers()
                if self.state == State.PENDING:
                    self.reset()
            self.current_session = row['Session']

        # PHASE 1: Build Asia Levels (Static structural anchors)
        if self.current_session == 'Asia':
            if row['High'] > self.asia_high:
                self.asia_high = row['High']
                self.origin_for_high = self.last_fractal_low # May be None if no fractal yet
            
            if row['Low'] < self.asia_low:
                self.asia_low = row['Low']
                self.origin_for_low = self.last_fractal_high

        # PHASE 2: London Sweep (Strictly during London session)
        elif self.current_session == 'London':
            if self.swept_side is None:
                if row['High'] > self.asia_high:
                    self.swept_side = 'HIGH'
                    self.sweep_extreme = row['High']
                elif row['Low'] < self.asia_low:
                    self.swept_side = 'LOW'
                    self.sweep_extreme = row['Low']
            else:
                # Update sweep extreme for invalidation tracking
                if self.swept_side == 'HIGH':
                    self.sweep_extreme = max(self.sweep_extreme, row['High'])
                else:
                    self.sweep_extreme = min(self.sweep_extreme, row['Low'])

        # PHASE 3: Confirmation (London or New York)
        # Check that we have a sweep AND that the origin levels aren't None (FIX)
        if self.swept_side is not None and self.current_session in ['London', 'New York']:
            
            # 1. NY Invalidation Guard
            if self.current_session == 'New York':
                if self.swept_side == 'HIGH' and row['High'] > self.sweep_extreme:
                    self._reset_daily_trackers() # Trend continued, not a fake move
                    return
                elif self.swept_side == 'LOW' and row['Low'] < self.sweep_extreme:
                    self._reset_daily_trackers()
                    return

            # 2. Safety Check & Confirmation
            if self.swept_side == 'HIGH' and self.origin_for_high is not None:
                if row['Close'] < self.origin_for_high:
                    self.mark_triggered(index, row)
                    
            elif self.swept_side == 'LOW' and self.origin_for_low is not None:
                if row['Close'] > self.origin_for_low:
                    self.mark_triggered(index, row)

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

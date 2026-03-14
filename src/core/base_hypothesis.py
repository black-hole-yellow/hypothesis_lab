from enum import Enum
import pandas as pd

class State(Enum):
    PENDING = "PENDING"       # Waiting for the trigger condition
    ACTIVE = "ACTIVE"         # Triggered, waiting for the body close / timeout
    COMPLETED = "COMPLETED"   # Event finished, ready to log

class BaseHypothesis:
    def __init__(self, hypothesis_id: str, description: str, variables: list):
        # 1. Identity
        self.hypothesis_id = hypothesis_id
        self.description = description
        self.variables = variables
        
        # 2. State Tracking
        self.state = State.PENDING
        self.result = None  # Will be True (Success) or False (Invalidated/Timeout)
        
        # 3. Data Capture
        self.trigger_time = None
        self.context_trend = None       # e.g., Norm_Slope
        self.context_volatility = None  # e.g., Body_ZScore

    def process_candle(self, index: pd.Timestamp, row: pd.Series):
        """
        The main router. Called by the engine for every new candle.
        """
        if self.state == State.PENDING:
            self.check_trigger(index, row)
        elif self.state == State.ACTIVE:
            self.update_state(index, row)

    def mark_triggered(self, index: pd.Timestamp, row: pd.Series):
        """
        Called inside check_trigger() when conditions are met.
        Captures the exact market context at the moment of the trigger.
        """
        self.state = State.ACTIVE
        self.trigger_time = index
        # Safely grab context if it exists in the dataframe
        self.context_trend = row.get('Norm_Slope', None)
        self.context_volatility = row.get('Body_ZScore', None)

    def mark_completed(self, outcome: bool):
        """
        Called inside update_state() when the hypothesis is proven or invalidated.
        """
        self.state = State.COMPLETED
        self.result = outcome

    def reset(self):
        """Resets the state machine for the next session/day."""
        self.state = State.PENDING
        self.result = None
        self.trigger_time = None
        self.context_trend = None
        self.context_volatility = None

    def get_csv_row(self) -> dict:
        """Packages the final data for the CSV Database."""
        return {
            "Hypothesis_ID": self.hypothesis_id,
            "Description": self.description,
            "Variables": str(self.variables),
            "Trigger_Time": self.trigger_time,
            "Context_Trend": self.context_trend,
            "Context_Volatility": self.context_volatility,
            "Result": self.result
        }

    # ==========================================
    # METHODS TO BE OVERRIDDEN BY YOUR SPECIFIC HYPOTHESES
    # ==========================================
    def check_trigger(self, index: pd.Timestamp, row: pd.Series):
        raise NotImplementedError("Must implement check_trigger() in child class")

    def update_state(self, index: pd.Timestamp, row: pd.Series):
        raise NotImplementedError("Must implement update_state() in child class")


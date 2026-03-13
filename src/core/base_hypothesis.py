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


# ==========================================
# TEST FUNCTION
# ==========================================
if __name__ == "__main__":
    # 1. Create a quick Mock Hypothesis to test the inheritance and state machine
    class MockHypothesis(BaseHypothesis):
        def __init__(self):
            super().__init__(
                hypothesis_id="MOCK_001", 
                description="Trigger if Close > 1.50, Complete if Close < 1.48", 
                variables=["Close"]
            )
            
        def check_trigger(self, index, row):
            if row['Close'] > 1.5000:
                print(f"[{index}] TRIGGERED: Close is {row['Close']}")
                self.mark_triggered(index, row)
                
        def update_state(self, index, row):
            if row['Close'] < 1.4800:
                print(f"[{index}] COMPLETED (True): Close dropped to {row['Close']}")
                self.mark_completed(True)
            elif (index - self.trigger_time).total_seconds() > 7200: # 2 hour timeout
                print(f"[{index}] COMPLETED (False): Time Out.")
                self.mark_completed(False)

    # 2. Generate dummy price feed
    mock_data = pd.DataFrame({
        'Close': [1.4900, 1.5050, 1.4950, 1.4750],
        'Norm_Slope': [0.1, 0.5, 0.2, -0.8],
        'Body_ZScore': [0.5, 2.1, 1.0, 3.5]
    }, index=pd.date_range("2026-02-10 10:00:00", periods=4, freq="1h"))

    print("--- Running Engine Test Loop ---")
    
    # 3. Instantiate and run the loop
    hypo = MockHypothesis()
    
    for idx, row in mock_data.iterrows():
        print(f"Processing {idx} | State before: {hypo.state.name}")
        hypo.process_candle(idx, row)
    
    print("\n--- Final CSV Output ---")
    print(hypo.get_csv_row())
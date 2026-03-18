import pandas as pd
from src.core.base_hypothesis import BaseHypothesis
from src.core.parser import SignalParser

class GenericJSONHypothesis(BaseHypothesis):
    def __init__(self, config: dict):
        # Dynamically name the hypothesis based on the JSON
        name = config.get("metadata", {}).get("name", "Unnamed_Generated_Hypothesis")
        super().__init__(name=name, config=config)
        
        # Initialize the parser with the parameters from the JSON
        self.parser = SignalParser(self.config.get("parameters", {}))
        
        # Extract the logic block for fast access during the loop
        self.logic = self.config.get("logic", {})
        self.entry_rules = self.logic.get("entry_rules", {})

    def evaluate_row(self, row: pd.Series, index: pd.Timestamp):
        """
        Dynamically scans the JSON entry rules. 
        If conditions are met, it fires a trigger.
        """
        # 1. Check for Long Triggers
        long_rules = self.entry_rules.get("long_trigger", [])
        if long_rules and self.parser.check_conditions(row, long_rules):
            self.triggers.append({'Datetime': index, 'Direction': 'Long'})
            return # Exit early if triggered to prevent double-firing

        # 2. Check for Short Triggers
        short_rules = self.entry_rules.get("short_trigger", [])
        if short_rules and self.parser.check_conditions(row, short_rules):
            self.triggers.append({'Datetime': index, 'Direction': 'Short'})
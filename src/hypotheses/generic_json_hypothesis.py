import pandas as pd
from src.core.base_hypothesis import BaseHypothesis
from src.core.parser import SignalParser

class GenericJSONHypothesis(BaseHypothesis):
    def __init__(self, config: dict):
        name = config.get("metadata", {}).get("name", "Unnamed_Generated_Hypothesis")
        # Ensure it inherits properly
        super().__init__(name=name, config=config)
        
        self.parser = SignalParser(self.config.get("parameters", {}))
        self.logic = self.config.get("logic", {})
        self.entry_rules = self.logic.get("entry_rules", {})

    def evaluate_row(self, row: pd.Series, index: pd.Timestamp):
        """
        Dynamically scans the JSON rules. 
        Logs the event and market context for research review.
        """
        # 1. Check for Bullish Profile Events
        long_rules = self.entry_rules.get("long_trigger", [])
        if long_rules and self.parser.check_conditions(row, long_rules):
            
            # Record for statistical evaluation (T-Stat, etc.)
            self.triggers.append({'Datetime': index, 'Direction': 'Long'})
            
            # RECORD FOR THE CSV AUDIT LOG
            log_entry = {
                'Datetime': index, 
                'Profile_Type': 'Bullish Setup Detected',
                'Close_Price': row['Close']
            }
            # Dynamically grab the features that triggered this so you can review them
            for rule in long_rules:
                feat = rule.get("feature")
                if feat: log_entry[feat] = row.get(feat)
                
            self.daily_logs.append(log_entry)
            return

        # 2. Check for Bearish Profile Events
        short_rules = self.entry_rules.get("short_trigger", [])
        if short_rules and self.parser.check_conditions(row, short_rules):
            
            self.triggers.append({'Datetime': index, 'Direction': 'Short'})
            
            log_entry = {
                'Datetime': index, 
                'Profile_Type': 'Bearish Setup Detected',
                'Close_Price': row['Close']
            }
            for rule in short_rules:
                feat = rule.get("feature")
                if feat: log_entry[feat] = row.get(feat)
                
            self.daily_logs.append(log_entry)
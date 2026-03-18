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
        Logs the event and market context in local Kyiv time for research review.
        """
        try:
            if index.tz is not None:
                ua_time = index.tz_convert('Europe/Kyiv').strftime('%Y-%m-%d %H:%M:%S')
            else:
                ua_time = index.tz_localize('UTC').tz_convert('Europe/Kyiv').strftime('%Y-%m-%d %H:%M:%S')
        except:
            ua_time = str(index) 

        # Extract the exact probability calculated by your DNA library
        trend_prob = row.get('HTF_Bullish_Prob', 'N/A')

        # 1. Check for Long Triggers
        long_rules = self.entry_rules.get("long_trigger", [])
        if long_rules and self.parser.check_conditions(row, long_rules):
            
            self.triggers.append({'Datetime': index, 'Direction': 'Long'})
            
            # Record numerical data instead of text strings
            log_entry = {
                'UA_Time': ua_time, 
                'Trigger_Direction': 'Long',
                'HTF_Bullish_Prob_%': trend_prob,  # <--- Exact percentage logged here!
                'Close_Price': row['Close']
            }
            for rule in long_rules:
                feat = rule.get("feature")
                if feat: log_entry[feat] = row.get(feat)
                
            self.daily_logs.append(log_entry)
            return

        # 2. Check for Short Triggers
        short_rules = self.entry_rules.get("short_trigger", [])
        if short_rules and self.parser.check_conditions(row, short_rules):
            
            self.triggers.append({'Datetime': index, 'Direction': 'Short'})
            
            log_entry = {
                'UA_Time': ua_time, 
                'Trigger_Direction': 'Short',
                'HTF_Bullish_Prob_%': trend_prob,  # <--- Exact percentage logged here!
                'Close_Price': row['Close']
            }
            for rule in short_rules:
                feat = rule.get("feature")
                if feat: log_entry[feat] = row.get(feat)
                
            self.daily_logs.append(log_entry)
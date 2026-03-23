import pandas as pd
from src.core.base_hypothesis import BaseHypothesis
from src.core.parser import SignalParser
from src.core.trade_tracker import TradeTracker

class GenericJSONHypothesis(BaseHypothesis):
    def __init__(self, config: dict):
        name = config.get("metadata", {}).get("name", "Unnamed_Generated_Hypothesis")
        super().__init__(name=name, config=config)
        
        self.parser = SignalParser(self.config.get("parameters", {}))
        
        # Strict Schema adoption
        self.logic = self.config.get("logic", {})
        self.execution = self.config.get("execution", {}) 
        
        self.filters = self.logic.get("filters", [])
        self.entry_rules = self.logic.get("entry_rules", {})
        
        # Initialize our new state manager
        self.tracker = TradeTracker(self.execution)

    def reset(self):
        """Overrides base reset to also wipe the tracker."""
        super().reset()
        self.tracker = TradeTracker(self.execution)

    def evaluate_row(self, row: pd.Series, index: pd.Timestamp):
        """Evaluates row, manages active trades, and generates signals."""
        
        # 1. Update daily limits
        self.tracker.update_day(index.date())
        
        # 2. Pre-evaluate signals (needed for potential signal-based exits)
        long_rules = self.entry_rules.get("long_trigger", [])
        short_rules = self.entry_rules.get("short_trigger", [])
        
        is_long = long_rules and self.parser.check_conditions(row, long_rules)
        is_short = short_rules and self.parser.check_conditions(row, short_rules)
        
        current_signal = 'Long' if is_long else ('Short' if is_short else None)

        # 3. Process existing trades (Age them by 1 bar, check exits)
        # The tracker modifies the dicts directly in memory.
        self.tracker.process_active_trades(row, counter_signal=current_signal)

        # 4. Check if we have a valid entry signal AND room to trade
        if current_signal and self.tracker.can_open_trade():
            
            # 5. Check Global Filters (Trend, Time, etc.)
            if self.filters and not self.parser.check_conditions(row, self.filters):
                return # Filters failed, ignore the signal
                
            # 6. Open the Trade
            trade_record = {
                'Datetime': index,
                'Direction': current_signal,
                'Entry_Price': row['Close']
            }
            
            self.triggers.append(trade_record)
            self.tracker.add_trade(trade_record) # Tracker takes over management
            
            # 7. Standardized Audit Logging
            try:
                ua_time = index.tz_convert('Europe/Kyiv').strftime('%Y-%m-%d %H:%M:%S') if index.tz else str(index)
            except:
                ua_time = str(index)
                
            log_entry = {
                'Datetime_Kyiv': ua_time,
                'Direction': current_signal,
                'Entry_Price': row.get('Close', 'N/A')
            }
            
            # Dynamically pull the exact features used for verification
            rules_used = self.filters + (long_rules if is_long else short_rules)
            for rule in rules_used:
                feat = rule.get("feature")
                if feat: log_entry[feat] = row.get(feat)
                
            self.daily_logs.append(log_entry)
import pandas as pd
from src.core.base_hypothesis import BaseHypothesis
from src.core.parser import SignalParser

class GenericJSONHypothesis(BaseHypothesis):
    def __init__(self, config: dict):
        name = config.get("metadata", {}).get("name", "Unnamed_Generated_Hypothesis")
        super().__init__(name=name, config=config)
        
        self.parser = SignalParser(self.config.get("parameters", {}))
        self.logic = self.config.get("logic", {})
        
        self.filters = self.logic.get("filters", [])
        self.entry_rules = self.logic.get("entry_rules", {})
        
        # --- НОВОЕ: Извлекаем правила исполнения ---
        self.execution_rules = self.logic.get("execution_rules", {
            "risk_reward_ratio": 2.0,       # Дефолт 1:2
            "max_trades_per_day": 999,      # Без лимита по дефолту
            "allow_resweep": False,
            "default_sl_atr_multiplier": 1.0
        })

    def evaluate_row(self, row: pd.Series, index: pd.Timestamp):
        """
        Evaluates the row against JSON rules.
        Creates a strict, standardized CSV output format for all hypotheses.
        """
        # 1. Format the Local Time
        try:
            if index.tz is not None:
                ua_time = index.tz_convert('Europe/Kyiv').strftime('%Y-%m-%d %H:%M:%S')
            else:
                ua_time = index.tz_localize('UTC').tz_convert('Europe/Kyiv').strftime('%Y-%m-%d %H:%M:%S')
        except:
            ua_time = str(index) 

        # 2. Check Global Filters
        if self.filters and not self.parser.check_conditions(row, self.filters):
            return 

        base_log = {
            'Datetime_Kyiv': ua_time,
            'Trend_%': row.get('HTF_Bullish_Prob', 'N/A'),
            'Close_Price': row.get('Close', 'N/A'),
            'Signal_Triggered': 'Yes', 
            'Direction': None
        }

        # 4. Check for LONG Triggers
        long_rules = self.entry_rules.get("long_trigger", [])
        if long_rules and self.parser.check_conditions(row, long_rules):
            self.triggers.append({'Datetime': index, 'Direction': 'Long'})
            log_entry = base_log.copy()
            log_entry['Direction'] = 'Long'
            for rule in self.filters + long_rules:
                feat = rule.get("feature")
                if feat: log_entry[feat] = row.get(feat)
            self.daily_logs.append(log_entry)
            return

        # 5. Check for SHORT Triggers
        short_rules = self.entry_rules.get("short_trigger", [])
        if short_rules and self.parser.check_conditions(row, short_rules):
            self.triggers.append({'Datetime': index, 'Direction': 'Short'})
            log_entry = base_log.copy()
            log_entry['Direction'] = 'Short'
            for rule in self.filters + short_rules:
                feat = rule.get("feature")
                if feat: log_entry[feat] = row.get(feat)
            self.daily_logs.append(log_entry)
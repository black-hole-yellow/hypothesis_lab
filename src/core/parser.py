import operator
import pandas as pd

class SignalParser:
    def __init__(self, parameters: dict):
        self.p = parameters
        
        # Map JSON string operators to Python's built-in math operators
        self.ops = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne
        }

    def _parse_value(self, val):
        """Translates 'param:xxx' into the actual number from the JSON config."""
        if isinstance(val, str) and "param:" in val:
            multiplier = -1 if val.startswith("-") else 1
            param_key = val.replace("-param:", "").replace("param:", "")
            return multiplier * self.p.get(param_key, 0.0)
        return val

    def evaluate_rule(self, row: pd.Series, rule: dict) -> bool:
        """Evaluates a single JSON rule against the current candle."""
        feature_value = row.get(rule.get("feature"))
        
        # If the feature doesn't exist in the data, the rule fails safely
        if feature_value is None or pd.isna(feature_value):
            return False
            
        target_value = self._parse_value(rule.get("value"))
        compare_function = self.ops.get(rule.get("operator"))
        
        if not compare_function:
            return False
            
        return compare_function(feature_value, target_value)

    def check_conditions(self, row: pd.Series, rule_list: list) -> bool:
        """Checks a list of rules. Returns True ONLY if ALL of them are met."""
        if not rule_list:
            return False
        return all(self.evaluate_rule(row, rule) for rule in rule_list)
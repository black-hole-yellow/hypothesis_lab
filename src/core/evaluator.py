import os
import pandas as pd
import numpy as np

class SignalEvaluator:
    def __init__(self, hypothesis: object):
        self.hypothesis_name = hypothesis.name
        self.triggers = hypothesis.triggers
        self.daily_logs = hypothesis.daily_logs

    def _export_audit_log(self):
        """Generates the standardized CSV for review."""
        if not self.daily_logs:
            return
            
        os.makedirs("output", exist_ok=True)
        safe_name = self.hypothesis_name.replace('/', '_').replace('\\', '_')
        audit_filename = f"output/{safe_name}_audit_log.csv"
        
        # Inject outcomes into the daily logs before saving
        for i, log in enumerate(self.daily_logs):
            if i < len(self.triggers):
                log['Outcome'] = self.triggers[i].get('Outcome', 'Unknown')
                log['Hold_Bars'] = self.triggers[i].get('Hold_Bars', 0)
                
        pd.DataFrame(self.daily_logs).to_csv(audit_filename, index=False)
        return audit_filename

    def calculate_metrics(self) -> dict:
        """Calculates statistical edge based on directional accuracy over time."""
        # Only evaluate closed trades
        completed = [t for t in self.triggers if t.get('Outcome') in ['Win', 'Loss']]
        freq = len(completed)
        
        # Save the audit log regardless of outcome
        self._export_audit_log()
        
        if freq < 10:
            # Not enough sample size for standard statistics, flag for manual review
            return {
                'Hypothesis': self.hypothesis_name, 
                'Status': 'REVIEW (Low Sample Size)', 
                'Frequency': freq
            }

        wins = sum(1 for t in completed if t['Outcome'] == 'Win')
        win_rate = wins / freq
        
        # Binomial Test vs 50% Null Hypothesis
        # Standard Error of proportion = sqrt( (p * (1-p)) / N ) 
        # For null hypothesis p = 0.5, variance is 0.25
        standard_error = np.sqrt(0.25 / freq)
        t_stat = (win_rate - 0.5) / standard_error if standard_error > 0 else 0

        # Pass criteria: T-Stat >= 2.0 (approx 95% confidence) AND Win Rate > 50%
        passed = t_stat >= 2.0 and win_rate > 0.5

        return {
            'Hypothesis': self.hypothesis_name,
            'Frequency': freq,
            'Win_Rate_%': round(win_rate * 100, 2),
            'Wins': wins,
            'Losses': freq - wins,
            'T_Stat': round(t_stat, 2),
            'Status': 'PASSED' if passed else 'FAILED'
        }
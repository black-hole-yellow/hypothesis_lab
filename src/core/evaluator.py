import os
import pandas as pd
import numpy as np

class SignalEvaluator:
    def __init__(self, hypothesis: object, df: pd.DataFrame = None):
        self.hypothesis_name = hypothesis.name
        self.triggers = hypothesis.triggers
        self.daily_logs = hypothesis.daily_logs
        self.df = df

    def _export_audit_log(self):
        """Generates the standardized CSV for review."""
        if not self.daily_logs:
            return
            
        os.makedirs("output", exist_ok=True)
        safe_name = self.hypothesis_name.replace('/', '_').replace('\\', '_')
        audit_filename = f"output/{safe_name}_audit_log.csv"
        
        for i, log in enumerate(self.daily_logs):
            if i < len(self.triggers):
                log['Outcome'] = self.triggers[i].get('Outcome', 'Unknown')
                log['Hold_Bars'] = self.triggers[i].get('Hold_Bars', 0)
                
        pd.DataFrame(self.daily_logs).to_csv(audit_filename, index=False)
        return audit_filename

    def calculate_metrics(self) -> dict:
        """Calculates statistical edge using a Dual-Path evaluation with Frequency tracking."""
        completed = [t for t in self.triggers if t.get('Outcome') in ['Win', 'Loss'] or t.get('Status') == 'Closed']
        total_freq = len(completed) # 1. Capture the "General Frequency"
        self._export_audit_log()
        
        if total_freq < 3:
            return {
                'Hypothesis': self.hypothesis_name, 
                'Status': 'REVIEW (Low Sample Size)', 
                'Frequency': total_freq,
                'Total_Frequency': total_freq
            }

        best_t_stat = -999.0
        best_win_rate = 0.0
        optimal_bars = 0
        best_wins = 0
        optimal_freq = total_freq # Default to total

        if self.df is not None:
            dt_to_idx = {dt: i for i, dt in enumerate(self.df.index)}
            
            for horizon in range(1, 25):
                horizon_wins = 0
                valid_trades = 0
                
                for t in completed:
                    entry_dt = t.get('Datetime')
                    direction = t.get('Direction', t.get('Type', t.get('Signal', 'Long')))
                    entry_price = t.get('Entry_Price')
                    
                    if entry_dt in dt_to_idx and entry_price is not None:
                        entry_idx = dt_to_idx[entry_dt]
                        exit_idx = entry_idx + horizon
                        
                        if exit_idx < len(self.df): # Ensure data exists for this specific hold
                            valid_trades += 1
                            exit_price = self.df['Close'].iloc[exit_idx]
                            
                            is_win = False
                            if isinstance(direction, str):
                                if direction.lower() == 'long' and exit_price > entry_price:
                                    is_win = True
                                elif direction.lower() == 'short' and exit_price < entry_price:
                                    is_win = True
                                    
                            if is_win:
                                horizon_wins += 1
                                
                if valid_trades >= 3:
                    h_win_rate = horizon_wins / valid_trades
                    h_se = np.sqrt(0.25 / valid_trades)
                    h_t_stat = (h_win_rate - 0.5) / h_se if h_se > 0 else 0
                    
                    if h_t_stat > best_t_stat:
                        best_t_stat = h_t_stat
                        best_win_rate = h_win_rate
                        optimal_bars = horizon
                        best_wins = horizon_wins
                        optimal_freq = valid_trades # Update to the valid count for this horizon

        # Dual-Path Evaluation
        is_rare_event = optimal_freq < 10
        if is_rare_event:
            passed = best_win_rate >= 0.75
            final_status = 'PASSED (Rare Event)' if passed else 'FAILED (Rare Event)'
        else:
            passed = best_t_stat >= 2.0 and best_win_rate > 0.5
            final_status = 'PASSED' if passed else 'FAILED'

        return {
            'Hypothesis': self.hypothesis_name,
            'Frequency': optimal_freq,        # Count used for T-Stat
            'Total_Frequency': total_freq,    # 2. General trigger count
            'Win_Rate_%': round(best_win_rate * 100, 2),
            'Wins': best_wins,
            'Losses': optimal_freq - best_wins,
            'T_Stat': round(best_t_stat, 2),
            'Optimal_Hold': optimal_bars,
            'Status': final_status
        }
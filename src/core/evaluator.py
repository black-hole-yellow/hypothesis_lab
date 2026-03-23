import pandas as pd
import numpy as np

class SignalEvaluator:
    def __init__(self, df: pd.DataFrame, triggers: list, hypothesis_name: str, target_col: str = 'Close'):
        self.df = df
        self.triggers = triggers
        self.hypothesis_name = hypothesis_name

    def calculate_metrics(self) -> dict:
        # Считаем только закрытые сделки
        completed = [t for t in self.triggers if t.get('Outcome') in ['Win', 'Loss']]
        freq = len(completed)
        
        if freq == 0: # Порог значимости отключен для макро; проверяется в batch_runner
            return {'Hypothesis': self.hypothesis_name, 'Status': 'FAILED (Not enough data)', 'Frequency': freq}

        wins = sum(1 for t in completed if t['Outcome'] == 'Win')
        win_rate = wins / freq
        
        # Expectancy: (WR * 2) - ((1-WR) * 1)
        ev_r = (win_rate * 2.0) - ((1 - win_rate) * 1.0)
        
        # T-Stat для R-multiples
        e_x2 = (win_rate * 4.0) + ((1 - win_rate) * 1.0)
        std_dev = np.sqrt(e_x2 - (ev_r ** 2)) if (e_x2 - (ev_r ** 2)) > 0 else 1e-8
        t_stat = (ev_r / std_dev) * np.sqrt(freq)

        return {
            'Hypothesis': self.hypothesis_name,
            'Frequency': freq,
            'Win_Rate_%': round(win_rate * 100, 2),
            'T_Stat': round(t_stat, 2),
            'Status': 'PASSED' if (t_stat >= 2.0 and ev_r > 0) else 'FAILED'
        }
import pandas as pd
import numpy as np
import os

class SignalEvaluator:
    def __init__(self, df: pd.DataFrame, triggers: list, hypothesis_name: str, target_col: str = 'Close'):
        self.df = df.copy()
        self.triggers = triggers
        self.hypothesis_name = hypothesis_name
        self.target_col = target_col

    def calculate_metrics(self) -> dict:
        # 1. Отбираем только закрытые сделки с известным результатом
        completed_trades = [t for t in self.triggers if t.get('Outcome') in ['Win', 'Loss']]
        freq = len(completed_trades)
        
        if freq < 2:
            return {
                'Hypothesis': self.hypothesis_name, 
                'Status': 'FAILED (Not enough data)', 
                'Frequency': freq,
                'Win_Rate_%': 0.0,
                'Expectancy_R': 0.0,
                'T_Stat': 0.0
            }

        # 2. Подсчет базовой статистики
        wins = sum(1 for t in completed_trades if t['Outcome'] == 'Win')
        losses = freq - wins
        win_rate = wins / freq
        loss_rate = losses / freq
        
        # 3. Расчет математического ожидания (Expectancy / EV) в R-множителях
        # Правило: Win = +2R, Loss = -1R
        ev_r = (win_rate * 2.0) - (loss_rate * 1.0)
        
        # 4. Расчет T-Statistic для R-множителей
        # Дисперсия (Variance) = E[X^2] - (E[X])^2
        # E[X^2] = win_rate * (2.0)^2 + loss_rate * (-1.0)^2
        e_x2 = (win_rate * 4.0) + (loss_rate * 1.0)
        variance = e_x2 - (ev_r ** 2)
        std_dev = np.sqrt(variance) if variance > 0 else 1e-8
        
        # T-Stat = (Среднее / Стандартное отклонение) * sqrt(N)
        t_stat = (ev_r / std_dev) * np.sqrt(freq) if freq > 0 else 0.0

        metrics = {
            'Hypothesis': self.hypothesis_name,
            'Frequency': freq,
            'Wins': wins,
            'Losses': losses,
            'Win_Rate_%': round(win_rate * 100, 2),
            'Expectancy_R': round(ev_r, 4),
            'T_Stat': round(t_stat, 2),
        }

        # 5. Гейт принятия решения (Decision Gate)
        # Нам нужно положительное матожидание (EV > 0) и стат. значимость (T-Stat > 2.0)
        if t_stat >= 2.0 and ev_r > 0.0:
            metrics['Status'] = 'PASSED'
        else:
            metrics['Status'] = 'FAILED (No Edge or Low Significance)'

        return metrics

def save_hypothesis_results(metrics: dict, filepath: str = "output/hypothesis_results.csv"):
    """Saves ONLY passed hypotheses to the registry."""
    if metrics.get('Status') != 'PASSED':
        print(f"[{metrics['Hypothesis']}] FAILED. Not saving to registry.")
        return

    df_new = pd.DataFrame([metrics])
    
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        # Check if we are updating an existing hypothesis
        if metrics['Hypothesis'] in df_existing['Hypothesis'].values:
            df_existing.set_index('Hypothesis', inplace=True)
            df_new.set_index('Hypothesis', inplace=True)
            df_existing.update(df_new)
            df_existing.reset_index(inplace=True)
            df_final = df_existing
        else:
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df_final.to_csv(filepath, index=False)
    print(f"✅ [{metrics['Hypothesis']}] PASSED and saved to {filepath}!")
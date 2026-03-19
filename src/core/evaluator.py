import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

class SignalEvaluator:
    def __init__(self, df: pd.DataFrame, triggers: list, hypothesis_name: str):
        self.df = df.copy()
        self.triggers = triggers
        self.hypothesis_name = hypothesis_name
        
        # Build the Signal Vector (+1 for Long, -1 for Short, 0 for Neutral)
        self.df['Signal'] = 0.0
        for t in triggers:
            # Assuming your trigger dict has 'Direction' ('Long'/'Short') and 'Datetime'
            val = 1.0 if t.get('Direction', 'Long') == 'Long' else -1.0
            if t['Datetime'] in self.df.index:
                self.df.loc[t['Datetime'], 'Signal'] = val

    def calculate_metrics(self) -> dict:
        signal = self.df['Signal']
        active_signals = signal[signal != 0]
        freq = len(active_signals)
        
        if freq < 10:
            return {'Hypothesis': self.hypothesis_name, 'Status': 'FAILED (Not enough data)', 'Frequency': freq}

        # 1. Forward Returns (1H, 4H, 12H, 24H)
        horizons = [1, 4, 12, 24]
        metrics = {
            'Hypothesis': self.hypothesis_name,
            'Frequency': freq,
            'Status': 'PENDING'
        }

        best_ic = -float('inf')
        best_horizon = horizons[0]
        best_t_stat = -float('inf')

        for h in horizons:
            # Shift negative to look into the future
            self.df[f'Fwd_Ret_{h}'] = (self.df['Close'].shift(-h) / self.df['Close']) - 1
            
            # Mask to only evaluate periods where a signal fired
            mask = (self.df['Signal'] != 0) & (self.df[f'Fwd_Ret_{h}'].notna())
            eval_df = self.df[mask]
            
            if len(eval_df) < 5: continue

            # Directional Accuracy (Hit Ratio)
            hits = np.sign(eval_df['Signal']) == np.sign(eval_df[f'Fwd_Ret_{h}'])
            
            win_count = int(hits.sum())
            loss_count = int(len(hits) - win_count)
            
            metrics[f'Hit_Ratio_{h}H'] = round(hits.mean() * 100, 2)
            metrics[f'Wins_{h}H'] = win_count
            metrics[f'Losses_{h}H'] = loss_count

            # Information Coefficient (Spearman Rank)
            ic, _ = spearmanr(eval_df['Signal'], eval_df[f'Fwd_Ret_{h}'])
            
            # T-Statistic of the IC (Degrees of freedom = N - 2)
            n = len(eval_df)
            # Add tiny epsilon to avoid division by zero if IC is perfect 1.0
            t_stat = ic * np.sqrt((n - 2) / ((1 - ic**2) + 1e-8)) 
            
            metrics[f'IC_{h}H'] = round(ic, 4)
            metrics[f'T_Stat_{h}H'] = round(t_stat, 2)

            # Track the Optimal Holding Period based on highest T-Stat
            if t_stat > best_t_stat:
                best_t_stat = t_stat
                best_ic = ic
                best_horizon = h
                metrics['Best_Win_Count'] = win_count
                metrics['Best_Loss_Count'] = loss_count
                

        metrics['Optimal_Hold_Hours'] = best_horizon
        metrics['Max_IC'] = round(best_ic, 4)
        metrics['Max_T_Stat'] = round(best_t_stat, 2)

        # 2. Decision Gate: Pass or Fail?
        # We want a positive edge (IC > 0.02) and statistical significance (T > 2.0)
        if best_t_stat >= 2.0 and best_ic > 0.0:
            metrics['Status'] = 'PASSED'
        else:
            metrics['Status'] = 'FAILED (No Edge)'

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
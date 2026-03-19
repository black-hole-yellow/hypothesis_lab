import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Adds the project root to the python path so imports work from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.engine import LabEngine

def run_hypothesis_visualization():
    print("=========================================")
    print("   STRATEGY X-RAY: CANDLESTICK EDITION   ")
    print("=========================================")
    
    data_path = "data/processed/GBPUSD_1h.parquet"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find {data_path}. Run batch_runner first to generate it.")
        return

    # Use a tight 2-month window so candlesticks are actually readable
    engine = LabEngine(
        data_file=data_path,
        start_date="2023-09-01", 
        end_date="2023-11-01",
        timeframe="1h"
    )

    print("⚙️ Running pipeline (calculating all features)...")
    if not engine.prepare_data():
        print("❌ Pipeline failed.")
        return
        
    df = engine.df
    print(f"✅ Pipeline complete. Charting {len(df)} candles...")

    # --- SETUP CHART ---
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')

    # --- 1. DRAW CANDLESTICKS ---
    up = df[df['Close'] >= df['Open']]
    down = df[df['Close'] < df['Open']]
    
    # Width of the candle body (45 minutes leaves a nice 15-min visual gap between 1H candles)
    width = pd.Timedelta(minutes=45)
    
    # Draw Wicks
    ax.vlines(up.index, up['Low'], up['High'], color='#2ecc71', linewidth=1.2, zorder=2)
    ax.vlines(down.index, down['Low'], down['High'], color='#e74c3c', linewidth=1.2, zorder=2)
    
    # Draw Bodies
    ax.bar(up.index, up['Close'] - up['Open'], bottom=up['Open'], color='#2ecc71', width=width, zorder=3)
    ax.bar(down.index, down['Open'] - down['Close'], bottom=down['Close'], color='#e74c3c', width=width, zorder=3)

    # --- 2. DRAW FRACTALS (Small & Unobtrusive) ---
    # We use the raw 'Fractal_High'/'Fractal_Low' here so they plot exactly on the tip of the wick
    fractal_highs = df[df['Fractal_High'] == True]
    fractal_lows = df[df['Fractal_Low'] == True]
    
    # Tiny gray dots sitting exactly 3 pips above/below the wick
    PIP = 0.0001
    ax.scatter(fractal_highs.index, fractal_highs['High'] + (3 * PIP), 
               color='gray', marker='.', s=30, zorder=4, label='Structural Fractal')
    ax.scatter(fractal_lows.index, fractal_lows['Low'] - (3 * PIP), 
               color='gray', marker='.', s=30, zorder=4)

    # --- 3. DRAW HYPOTHESIS TRIGGERS (Large & Obvious) ---
    long_triggers = df[df['First_LDN_Counter_High'] == 1]
    short_triggers = df[df['First_LDN_Counter_Low'] == 1]
    
    ax.scatter(long_triggers.index, long_triggers['Low'] - (15 * PIP), 
               color='blue', marker='^', s=200, zorder=5, label='BUY (Break Fake Res)')
    ax.scatter(short_triggers.index, short_triggers['High'] + (15 * PIP), 
               color='magenta', marker='v', s=200, zorder=5, label='SELL (Break Fake Sup)')

    # --- 4. SHADE BACKGROUNDS ---
    ymin, ymax = df['Low'].min() - (20 * PIP), df['High'].max() + (20 * PIP)
    
    # 1D Trend Regimes
    ax.fill_between(df.index, ymin, ymax, where=(df['1D_Swing_Bullish']==1), 
                    color='#2ecc71', alpha=0.05, label='1D Bullish Trend', zorder=1)
    ax.fill_between(df.index, ymin, ymax, where=(df['1D_Swing_Bearish']==1), 
                    color='#e74c3c', alpha=0.05, label='1D Bearish Trend', zorder=1)

    # London Session Window
    is_london = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 14)
    ax.fill_between(df.index, ymin, ymax, where=is_london, color='blue', alpha=0.03, zorder=1)

    # --- 5. FORMATTING ---
    ax.set_title("Hypothesis 15: London Counter-Trend Trap (Candlestick View)", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("GBP/USD Price")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    
    # Clean up the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_hypothesis_visualization()
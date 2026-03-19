import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Adds the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.data_loader import load_parquet_data

def run_volatility_event_study():
    print("=========================================")
    print("   MACRO VOLATILITY PROFILER: ELECTIONS  ")
    print("=========================================")
    
    # 1. Загрузка данных
    df = load_parquet_data("data/processed/GBPUSD_1h.parquet")
    if df.empty:
        print("Data not found.")
        return
        
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # 2. Рассчитываем Реализованную Волатильность (4H Rolling ATR в пипсах)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['Realized_Vol'] = df['TR'].rolling(window=4).mean() * 10000 # В пипсах

    # 3. Загрузка реестра событий
    with open("data/macro_events.json", 'r') as f:
        macro_data = json.load(f)
        
    elections = macro_data.get('Elections', [])
    if not elections:
        print("No elections found in macro_events.json")
        return

    # 4. Извлечение окна T-72h до T+72h
    window_hours = 72
    profiles = []

    for event in elections:
        dt = pd.to_datetime(event['start_date'])
        if dt.tz is None:
            dt = dt.tz_localize('UTC')
        else:
            dt = dt.tz_convert('UTC')
            
        t_zero = dt.floor('h')
        
        if t_zero in df.index:
            t_zero_idx = df.index.get_loc(t_zero)
            
            # Проверяем, есть ли достаточно данных вокруг события
            if t_zero_idx >= window_hours and (t_zero_idx + window_hours) < len(df):
                # Вырезаем окно волатильности
                vol_window = df['Realized_Vol'].iloc[t_zero_idx - window_hours : t_zero_idx + window_hours + 1].values
                profiles.append(vol_window)
                print(f"✅ Профиль загружен: {event['name']}")

    if not profiles:
        print("Не удалось извлечь профили событий.")
        return

    # 5. Усреднение (Находим среднюю волатильность по всем выборам)
    avg_profile = np.mean(profiles, axis=0)
    x_axis = np.arange(-window_hours, window_hours + 1)

    # 6. Визуализация (Рисуем Institutional Chart)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')

    ax.plot(x_axis, avg_profile, color='#e74c3c', linewidth=2.5, label='Average Realized Volatility (ATR Pips)')
    
    # Отмечаем T=0 (Экзит-поллы)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='T=0 (Exit Polls Released)')
    
    # Отмечаем T-48h
    ax.axvline(x=-48, color='blue', linestyle=':', linewidth=1.5, label='T-48h (Peak Uncertainty)')

    ax.set_title("GBP/USD Volatility Crush: UK Elections Event Study", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Hours Relative to Exit Polls (T=0)", fontsize=12)
    ax.set_ylabel("Volatility Magnitude (Pips)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_volatility_event_study()
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.data_loader import load_and_prep_data
from src.library.features import add_volatility_zscore, add_normalized_slope
from src.library.htf_features import add_previous_boundaries
from src.utils.macro_registry import load_macro_events

PIP = 0.0001

def calculate_macro_facts():
    print("--- Starting Macro-Sentiment Analyzer ---")
    
    # 1. Load Data & Events
    data_file = "data/gbpusd_data.csv"
    df = load_and_prep_data(data_file, '2025-01-01', '2026-02-27', '1h')
    
    # Calculate Base Technical DNA across entire history to ensure accuracy
    df = add_previous_boundaries(df)
    df = add_volatility_zscore(df, lookback=50)
    df = add_normalized_slope(df, lookback=20, atr_lookback=14)
    
    events = load_macro_events()
    if not events:
        return

    # Container for all our extracted facts
    event_facts = []

    # 2. Slicing Engine
    for event in events:
        # Buffer window: 3 Days before and after
        window_start = event['start_date'] - pd.Timedelta(days=3)
        window_end = event['end_date'] + pd.Timedelta(days=3)
        
        # Extract the Micro-Dataset for this event
        window_df = df.loc[window_start:window_end].copy()
        
        if window_df.empty:
            print(f"Skipping {event['event_id']} (No data in range)")
            continue

        # =======================================================
        # 3. FACT EXTRACTOR (Add new facts to this block in the future)
        # =======================================================
        
        # Fact A: Average Event Volatility (Z-Score)
        avg_volatility = window_df['Body_ZScore'].mean()
        max_volatility = window_df['Body_ZScore'].max()
        
        # Fact B: Trend Signature (Slope)
        avg_slope = window_df['Norm_Slope'].mean()
        
        # Fact C: 4H Reaction to HTF Wick Sweeps
        # Logic: Find every time High >= PDH/PWH or Low <= PDL/PWL. Check price 4 hours later.
        htf_reactions = []
        
        # Iterate up to the last 4 hours of the window
        for i in range(len(window_df) - 4):
            current_idx = window_df.index[i]
            row = window_df.iloc[i]
            
            # The price exactly 4 hours after the interaction
            price_4h_later = window_df.iloc[i + 4]['Close']
            
            # Bearish Sweep Reaction (Swept High, expecting drop)
            if pd.notna(row['PDH']) and row['High'] >= row['PDH']:
                reaction_pips = (row['PDH'] - price_4h_later) / PIP # Positive = Successful drop
                htf_reactions.append(reaction_pips)
            elif pd.notna(row['PWH']) and row['High'] >= row['PWH']:
                reaction_pips = (row['PWH'] - price_4h_later) / PIP
                htf_reactions.append(reaction_pips)
                
            # Bullish Sweep Reaction (Swept Low, expecting rally)
            if pd.notna(row['PDL']) and row['Low'] <= row['PDL']:
                reaction_pips = (price_4h_later - row['PDL']) / PIP # Positive = Successful rally
                htf_reactions.append(reaction_pips)
            elif pd.notna(row['PWL']) and row['Low'] <= row['PWL']:
                reaction_pips = (price_4h_later - row['PWL']) / PIP
                htf_reactions.append(reaction_pips)

        avg_4h_reaction = sum(htf_reactions) / len(htf_reactions) if htf_reactions else 0.0
        
        # Compile Facts for this specific event
        event_facts.append({
            'Event_ID': event['event_id'],
            'Category': event['category'],
            'Max_Vol_ZScore': round(max_volatility, 2),
            'Avg_Trend_Slope': round(avg_slope, 4),
            'Total_HTF_Sweeps': len(htf_reactions),
            'Avg_4h_Reaction_Pips': round(avg_4h_reaction, 1)
        })

    # =======================================================
    # 4. STATISTICAL AGGREGATION
    # =======================================================
    facts_df = pd.DataFrame(event_facts)
    
    # Export individual event facts
    os.makedirs("output", exist_ok=True)
    facts_df.to_csv("output/macro_event_facts.csv", index=False)
    
    print("\n--- MACRO REPORT GENERATED ---")
    print("Individual Event Signatures saved to 'output/macro_event_facts.csv'\n")
    
    # Group by Category to find the "Macro Pattern"
    print("--- CATEGORY FACT SHEET ---")
    category_summary = facts_df.groupby('Category').agg({
        'Event_ID': 'count',
        'Max_Vol_ZScore': 'mean',
        'Avg_Trend_Slope': 'mean',
        'Avg_4h_Reaction_Pips': 'mean'
    }).rename(columns={'Event_ID': 'Events_Analyzed'})
    
    print(category_summary.to_string())

if __name__ == "__main__":
    calculate_macro_facts()
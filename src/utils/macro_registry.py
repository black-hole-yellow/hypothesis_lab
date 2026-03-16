import json
import os
import pandas as pd

def load_macro_events(filepath: str = "data/macro_events.json") -> list:
    """
    Loads the optimized macro event registry.
    Automatically assigns categories and generates Event IDs based on dates.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Macro registry not found at {filepath}")
        return []
        
    with open(filepath, 'r') as file:
        data = json.load(file)
        
    events = []
    # Loop through the Categories (e.g., "US_Presidential_Election")
    for category, event_list in data.items():
        # Loop through the list of events under that category
        for details in event_list:
            event = details.copy()
            event['category'] = category
            
            # Convert strings to Timestamps
            start_dt = pd.to_datetime(event['start_date'])
            end_dt = pd.to_datetime(event['end_date'])
            
            # Auto-generate a clean ID like "US_Presidential_Election_20161108"
            date_str = start_dt.strftime('%Y%m%d')
            event['event_id'] = f"{category}_{date_str}"
            
            event['start_date'] = start_dt
            event['end_date'] = end_dt
            
            events.append(event)
            
    return events
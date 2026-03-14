import json
import os
import pandas as pd

def load_macro_events(filepath: str = "data/macro_events.json") -> list:
    """
    Loads the macro event registry and converts dates to pandas Timestamps.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Macro registry not found at {filepath}")
        return []
        
    with open(filepath, 'r') as file:
        data = json.load(file)
        
    events = []
    for event_id, details in data.items():
        event = details.copy()
        event['event_id'] = event_id
        # Convert strings to pandas Timestamps for easy slicing
        event['start_date'] = pd.to_datetime(event['start_date'])
        event['end_date'] = pd.to_datetime(event['end_date'])
        events.append(event)
        
    return events
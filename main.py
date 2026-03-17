import os
from src.core.engine import LabEngine
from src.hypotheses.london_fake_move import LondonFakeMove

def main():
    # --- RESEARCH PARAMETERS (Easy to tweak here!) ---
    research_config = {
        "strategy_name": "London_Pro_Trend_Sweep_v1",
        "universe": ["GBPUSD"],
        "data_dependencies": {
            "resolution": "1h",
            "required_features": ["HTF_Bullish_Prob", "Fractal_Up", "Fractal_Down"]
        },
        "parameters": {
            "asian_start_hour": 18,
            "asian_end_hour": 3,
            "eval_start_hour": 3,
            "eval_end_hour": 16,
            "bullish_trend_threshold": 50.0
        }
    }

    engine = LabEngine(
        data_file="data/gbpusd_data.csv",
        start_date="2015-01-01",
        end_date="2026-02-27",
        timeframe=research_config["data_dependencies"]["resolution"]
    )

    if not engine.prepare_data():
        return

    active_hypothesis = LondonFakeMove(config=research_config)

    engine.run_hypothesis(active_hypothesis)
    engine.evaluate(active_hypothesis)

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("configs/production", exist_ok=True) # Prepare the production folder
    main()
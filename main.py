import os
from src.core.engine import LabEngine

# IMPORT YOUR HYPOTHESES HERE
from src.hypotheses.london_fake_move import LondonFakeMove

def main():
    # 1. Initialize the Engine
    engine = LabEngine(
        data_file="data/gbpusd_data.csv",
        start_date="2015-01-01",
        end_date="2026-02-27",
        timeframe="1h"
    )

    # 2. Build the dataset
    if not engine.prepare_data():
        return

    # 3. SELECT YOUR HYPOTHESIS
    active_hypothesis = LondonFakeMove(name="London_Sweep_Daily_Profile")

    # 4. Run and Evaluate
    engine.run_hypothesis(active_hypothesis)
    engine.evaluate(active_hypothesis)

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    main()
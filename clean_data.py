import argparse
from data.data_polisher import DataPolisher

import sys
from pathlib import Path

# Force Python to recognize the project root
sys.path.append(str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Process and resample raw trading data.")
    
    # Defaults are set to your current Forex setup
    parser.add_argument('--file', type=str, default="data/gbpusd_data.csv", help="Path to raw CSV")
    parser.add_argument('--symbol', type=str, default="GBPUSD", help="Symbol name")
    parser.add_argument('--asset', type=str, default="forex", choices=["forex", "crypto", "equities"], help="Asset class behavior")
    parser.add_argument('--tz', type=str, default="Etc/GMT+5", help="Source timezone (e.g., UTC, Etc/GMT+5)")
    parser.add_argument('--sep', type=str, default="\t", help="CSV separator (e.g., \\t or ,)")
    
    args = parser.parse_args()

    polisher = DataPolisher(
        raw_csv_path=args.file,
        source_tz=args.tz,
        asset_class=args.asset
    )
    
    polisher.process_pipeline(symbol=args.symbol, separator=args.sep)

if __name__ == "__main__":
    main()
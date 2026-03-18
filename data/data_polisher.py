import pandas as pd
import os
import pyarrow

class DataPolisher:
    def __init__(self, raw_csv_path: str, output_dir: str = "data/processed/"):
        self.raw_path = raw_csv_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None

    def process_pipeline(self, symbol: str):
        print(f"🧹 Starting Data Polishing Pipeline for {symbol}...")
        
        self._load_and_fix_headers()
        self._normalize_timezones()
        self._force_continuous_index()
        self._filter_fx_weekends()
        
        # Save base 15m to Parquet
        base_file = os.path.join(self.output_dir, f"{symbol}_15m.parquet")
        self.df.to_parquet(base_file, engine='pyarrow')
        print(f"✅ Saved pristine 15m data to {base_file}")

        # Generate and save multiple timeframes
        self._resample_and_save(symbol, '1h')
        self._resample_and_save(symbol, '4h')
        self._resample_and_save(symbol, '1D') # FX aligned Daily
        
        print("🚀 Data Polishing Complete.")

    def _load_and_fix_headers(self):
        """Forces correct headers, handles tab-separators, and injects missing volume."""
        print("  -> Loading data and enforcing headers...")
        
        # 1. Use sep='\t' since the data is tab-separated, not comma-separated.
        # We read the first row to check if headers exist.
        sample = pd.read_csv(self.raw_path, sep='\t', nrows=1)
        
        if not isinstance(sample.iloc[0, 0], str):
            self.df = pd.read_csv(self.raw_path, sep='\t', header=None)
        else:
            self.df = pd.read_csv(self.raw_path, sep='\t')

        # 2. Count the columns to handle missing Volume data
        num_cols = len(self.df.columns)
        
        if num_cols >= 6:
            self.df = self.df.iloc[:, :6] # Keep first 6
            self.df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        elif num_cols == 5:
            # If volume is missing from the data provider, we inject it as 0
            self.df = self.df.iloc[:, :5]
            self.df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close']
            self.df['Volume'] = 0.0  
            print("     ! Notice: Data only had 5 columns. Injected dummy Volume column.")
        else:
            raise ValueError(f"❌ Data has an unexpected number of columns: {num_cols}. Expected 5 or 6.")
            
        # 3. Format the index securely
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        self.df.set_index('Datetime', inplace=True)

    def _normalize_timezones(self):
        """Converts EST (without DST) to strict UTC."""
        print("  -> Normalizing timezones to strict UTC...")
        # EST without DST is strictly UTC-5 all year round. 
        # In Pandas, this is represented by 'Etc/GMT+5' (Note: signs are inverted in POSIX)
        if self.df.index.tz is None:
            self.df.index = self.df.index.tz_localize('Etc/GMT+5')
        
        self.df.index = self.df.index.tz_convert('UTC')

    def _force_continuous_index(self):
        """Forces a perfect 15m grid. Forward-fills missing closes."""
        print("  -> Forcing continuous 15m index and patching missing bars...")
        
        # Create a mathematically perfect time grid from start to finish
        full_index = pd.date_range(start=self.df.index.min(), 
                                   end=self.df.index.max(), 
                                   freq='15min', 
                                   tz='UTC')
        
        # Reindex to force the grid. Missing bars become NaN.
        self.df = self.df.reindex(full_index)

        # 1. Forward-fill the Close price (the last known value)
        self.df['Close'] = self.df['Close'].ffill()
        
        # 2. For missing bars, Open, High, and Low simply equal the flatlined Close
        for col in ['Open', 'High', 'Low']:
            self.df[col] = self.df[col].fillna(self.df['Close'])
            
        # 3. Fill missing Volume with 0 (no trading happened)
        self.df['Volume'] = self.df['Volume'].fillna(0)

    def _filter_fx_weekends(self):
        """Removes Friday 22:00 UTC to Sunday 22:00 UTC (FX Dead Zone)."""
        print("  -> Filtering out FX weekend ghost bars...")
        
        # Pandas dayofweek: Monday=0, Sunday=6
        # Drop if: It's Saturday (5) OR (Friday after 22:00) OR (Sunday before 22:00)
        is_weekend = (
            (self.df.index.dayofweek == 5) | 
            ((self.df.index.dayofweek == 4) & (self.df.index.hour >= 22)) | 
            ((self.df.index.dayofweek == 6) & (self.df.index.hour < 22))
        )
        self.df = self.df[~is_weekend]

    def _resample_and_save(self, symbol: str, timeframe: str):
        """Resamples strictly and saves to Parquet."""
        print(f"  -> Resampling to {timeframe}...")
        
        ohlcv_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

        # Handle Institutional FX Daily Close (17:00 EST / 22:00 UTC)
        if timeframe == '1D':
            # Offset the daily resample by 22 hours so the daily candle closes exactly at NY Close
            resampled_df = self.df.resample('1D', offset='22h').agg(ohlcv_dict)
        else:
            resampled_df = self.df.resample(timeframe).agg(ohlcv_dict)
            
        # Drop any NaNs generated by resampling over the weekend gap
        resampled_df.dropna(inplace=True)

        out_path = os.path.join(self.output_dir, f"{symbol}_{timeframe}.parquet")
        resampled_df.to_parquet(out_path, engine='pyarrow')
        print(f"     Saved {timeframe} to {out_path}")
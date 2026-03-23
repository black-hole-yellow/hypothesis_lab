import pandas as pd
import os

class DataPolisher:
    def __init__(self, raw_csv_path: str, output_dir: str = "data/processed/", 
                 source_tz: str = 'Etc/GMT+5', asset_class: str = 'forex'):
        self.raw_path = raw_csv_path
        self.output_dir = output_dir
        self.source_tz = source_tz
        self.asset_class = asset_class.lower()
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None

    def process_pipeline(self, symbol: str, separator: str = '\t'):
        print(f"🧹 Starting Data Polishing Pipeline for {symbol} ({self.asset_class.upper()})...")
        
        self._load_and_fix_headers(separator)
        self._normalize_timezones()
        self._force_continuous_index()
        
        if self.asset_class == 'forex':
            self._filter_fx_weekends()
        
        # Save base 15m to Parquet
        base_file = os.path.join(self.output_dir, f"{symbol}_15m.parquet")
        self.df.to_parquet(base_file, engine='pyarrow')
        print(f"✅ Saved pristine 15m data to {base_file}")

        # Generate and save multiple timeframes
        self._resample_and_save(symbol, '1h')
        self._resample_and_save(symbol, '4h')
        self._resample_and_save(symbol, '1D') 
        
        print("🚀 Data Polishing Complete.")

    def _load_and_fix_headers(self, separator: str):
        """Robust header fixing. Sniffs columns and injects volume safely."""
        print("  -> Loading data and enforcing headers...")
        
        # Read a sample to check if headers exist in the raw file
        sample = pd.read_csv(self.raw_path, sep=separator, nrows=1)
        header_row = None if not isinstance(sample.iloc[0, 0], str) else 'infer'
        
        self.df = pd.read_csv(self.raw_path, sep=separator, header=header_row)

        # Standardize columns based on count
        cols = list(self.df.columns)
        if len(cols) >= 6:
            self.df = self.df.iloc[:, :6]
            self.df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        elif len(cols) == 5:
            self.df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close']
            self.df['Volume'] = 0.0  
            print("     ! Notice: Injected dummy Volume column.")
        else:
            raise ValueError(f"❌ Unexpected number of columns: {len(cols)}. Expected 5 or 6.")
            
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        self.df.set_index('Datetime', inplace=True)

    def _normalize_timezones(self):
        """Converts from the specified source timezone to strict UTC."""
        print(f"  -> Normalizing timezones from {self.source_tz} to UTC...")
        if self.df.index.tz is None:
            self.df.index = self.df.index.tz_localize(self.source_tz)
        self.df.index = self.df.index.tz_convert('UTC')

    def _force_continuous_index(self):
        """Forces a perfect grid. Forward-fills missing closes."""
        print("  -> Forcing continuous 15m index and patching missing bars...")
        full_index = pd.date_range(start=self.df.index.min(), 
                                   end=self.df.index.max(), 
                                   freq='15min', 
                                   tz='UTC')
        
        self.df = self.df.reindex(full_index)
        self.df['Close'] = self.df['Close'].ffill()
        
        for col in ['Open', 'High', 'Low']:
            self.df[col] = self.df[col].fillna(self.df['Close'])
            
        self.df['Volume'] = self.df['Volume'].fillna(0)

    def _filter_fx_weekends(self):
        """Removes Friday 22:00 UTC to Sunday 22:00 UTC (FX Dead Zone)."""
        print("  -> Filtering out FX weekend ghost bars...")
        is_weekend = (
            (self.df.index.dayofweek == 5) | 
            ((self.df.index.dayofweek == 4) & (self.df.index.hour >= 22)) | 
            ((self.df.index.dayofweek == 6) & (self.df.index.hour < 22))
        )
        self.df = self.df[~is_weekend]

    def _resample_and_save(self, symbol: str, timeframe: str):
        """Resamples strictly and saves to Parquet."""
        print(f"  -> Resampling to {timeframe}...")
        ohlcv_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}

        # Apply specific offset for Daily candles based on asset class
        if timeframe == '1D' and self.asset_class == 'forex':
            resampled_df = self.df.resample('1D', offset='22h').agg(ohlcv_dict)
        else:
            resampled_df = self.df.resample(timeframe).agg(ohlcv_dict)
            
        resampled_df.dropna(inplace=True)
        out_path = os.path.join(self.output_dir, f"{symbol}_{timeframe}.parquet")
        resampled_df.to_parquet(out_path, engine='pyarrow')
        print(f"     Saved {timeframe} to {out_path}")
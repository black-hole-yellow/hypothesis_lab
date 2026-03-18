from data.data_polisher import DataPolisher

polisher = DataPolisher(raw_csv_path="data/gbpusd_data.csv")
polisher.process_pipeline(symbol="GBPUSD")
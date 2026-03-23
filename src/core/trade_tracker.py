import pandas as pd

class TradeTracker:
    def __init__(self, execution_rules: dict):
        self.max_hold_bars = execution_rules.get("max_hold_bars", 8)
        self.allow_signal_exit = execution_rules.get("allow_signal_exit", False)
        self.max_trades_per_day = execution_rules.get("max_trades_per_day", 1)
        
        self.active_trades = []
        self.trades_opened_today = 0
        self.current_day = None

    def update_day(self, current_date):
        """Resets daily limits if a new day starts."""
        if current_date != self.current_day:
            self.current_day = current_date
            self.trades_opened_today = 0

    def can_open_trade(self) -> bool:
        """Circuit breaker for overtrading."""
        return self.trades_opened_today < self.max_trades_per_day

    def add_trade(self, trade_info: dict):
        """Registers a new trade."""
        trade_info['Hold_Bars'] = 0
        trade_info['Status'] = 'Active'
        trade_info['Outcome'] = 'Pending'
        self.active_trades.append(trade_info)
        self.trades_opened_today += 1

    def process_active_trades(self, row: pd.Series, counter_signal: str = None):
        """
        Ages active trades and closes them based on time or counter-signals.
        Returns a list of trades that were closed on this bar.
        """
        still_active = []
        just_closed = []

        for trade in self.active_trades:
            trade['Hold_Bars'] += 1
            
            # Check Exit Conditions
            time_exit = trade['Hold_Bars'] >= self.max_hold_bars
            signal_exit = self.allow_signal_exit and counter_signal and trade['Direction'] != counter_signal
            
            if time_exit or signal_exit:
                trade['Status'] = 'Closed'
                trade['Exit_Price'] = row['Close']
                
                # Basic direction check for forward returns (Evaluator does the exact math later)
                if trade['Direction'] == 'Long':
                    trade['Outcome'] = 'Win' if row['Close'] > trade['Entry_Price'] else 'Loss'
                else:
                    trade['Outcome'] = 'Win' if row['Close'] < trade['Entry_Price'] else 'Loss'
                    
                just_closed.append(trade)
            else:
                still_active.append(trade)

        self.active_trades = still_active
        return just_closed
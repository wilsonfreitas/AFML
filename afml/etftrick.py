
import pandas as pd


class FuturesRollETFTrick:
    def __init__(self, df):
        self.data_index = df.index
        df['pinpoint'] = ~(df['symbol'] == df['symbol'].shift(1))
        df = df.reset_index()
        df['Diff_close'] = df['close'].diff()
        df['Diff_close_open'] = df['close'] - df['open']
        df['H_part'] = 1/df['open'].shift(-1)
        self.prev_h = 1
        self.prev_k = 1
        _ = zip(df.H_part, df.Diff_close, df.Diff_close_open, df.pinpoint)
        df['K'] = [self.process_row(x, y, z, w) for x, y, z, w in _]
        self.data = df['K'].values

    @property
    def series(self):
        return pd.Series(self.data, index=self.data_index)

    def process_row(self, h_part, diff_close, diff_open_close, pinpoint):
        if pinpoint:
            h = self.prev_k*h_part
            delta = diff_open_close
        else:
            h = self.prev_h
            delta = diff_close
        k = self.prev_k + h*delta
        self.prev_h = h
        self.prev_k = k
        return k

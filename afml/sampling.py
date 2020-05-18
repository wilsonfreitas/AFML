
import datetime
import pandas as pd
import numpy as np


def compute_bar(df):
    if len(df) > 0:
        q = df['volume']
        p = df['price']
        vwap = np.cumsum(p * q) / np.cumsum(q)
        last = df.iloc[-1]
        first = df.iloc[0]
        return pd.DataFrame({
            'time': last['time'],
            'open': first['price'],
            'high': df['price'].max(),
            'low': df['price'].min(),
            'close': last['price'],
            'volume': df['volume'].sum(),
            'vwap': vwap.iloc[-1],
            'duration': (last['time'] - first['time']).total_seconds(),
            'ticks': len(df),
        }, index=[0])
    else:
        return None


def create_bars(df, grp, func):
    df_grp = df.groupby(grp)
    df_trans = df_grp.apply(func)
    return df_trans.reset_index(drop=True)


def tick_rule(price):
    z = {'prev_val': np.NaN}

    def f(x, y):
        if y:
            return z['prev_val']
        else:
            z['prev_val'] = x
            return x
    diff = price.diff()
    imbalance_raw = np.abs(diff)/diff
    imb = [f(x, y) for x, y in zip(imbalance_raw, imbalance_raw.isnull())]
    return pd.Series(imb, index=price.index)


class TickTEvents:
    def __init__(self, imbalance, expected_window, expected_imbalance, alpha):
        self.ticks = 0
        self.avg_ticks = expected_window
        self.avg_imbalance = expected_imbalance
        self.threshold = self.avg_ticks*abs(self.avg_imbalance)
        self._statistics = []
        self._thresholds = None
        self._events = None
        self.imbalance = imbalance
        self.alpha = alpha
        self.get_events()

    @property
    def statistics(self):
        if len(self._statistics) == 0:
            raise ValueError('statistics empty. get_events must be executed.')
        if len(self._statistics) != len(self.imbalance.index):
            raise ValueError('Statistics and imbalance index mismatch.')
        else:
            return pd.Series(self._statistics, index=self.imbalance.index)

    @property
    def thresholds(self):
        if self._thresholds is None:
            raise ValueError('threshold empty. get_events must be executed.')
        else:
            return self._thresholds

    @property
    def events(self):
        if self._events is None:
            raise ValueError('events empty. get_events must be executed.')
        else:
            return self._events

    def get_events(self):
        imb = self.imbalance.copy()
        imb[imb.isnull()] = 0
        _ = zip(imb.index, imb)
        lidx = self.imbalance.index[-1]  # last index
        self._statistics = []
        res = [self.process_row(x, y, lidx, self.alpha) for x, y in _]
        res = [row for row in res if row is not None]
        res = pd.DataFrame(res, columns=['tevent', 'threshold'])
        res = res.set_index('tevent')
        self._thresholds = res['threshold']
        self._events = res.index[:-1]

    def process_row(self, index, imbalance, last, alpha):
        raise NotImplementedError()


class DollarVolumeTEvents(TickTEvents):
    def __init__(self, imbalance, volume, expected_window, expected_imbalance,
                 alpha):
        self.volume = volume
        super().__init__(imbalance, expected_window, expected_imbalance, alpha)

    def get_events(self):
        imb = self.imbalance.copy()
        imb[imb.isnull()] = 0
        _ = zip(imb.index, imb, self.volume)
        lidx = self.imbalance.index[-1]  # last index
        self._statistics = []
        res = [self.process_row(x, y, z, lidx, self.alpha) for x, y, z in _]
        res = [row for row in res if row is not None]
        res = pd.DataFrame(res, columns=['tevent', 'threshold'])
        res = res.set_index('tevent')
        self._thresholds = res['threshold']
        self._events = res.index[:-1]


class TickImbalanceEvents(TickTEvents):
    def __init__(self, imbalance, expected_window, expected_imbalance, alpha):
        self.cum_imbalance = 0
        super().__init__(imbalance, expected_window, expected_imbalance, alpha)

    def process_row(self, index, imbalance, last, alpha):
        # _imbalance = 0 if imbalance_isnull else imbalance
        self.ticks += 1
        self.avg_imbalance = alpha*imbalance + (1-alpha)*self.avg_imbalance
        self.cum_imbalance += imbalance
        self._statistics.append(abs(self.cum_imbalance))
        res = None
        if self._statistics[-1] >= self.threshold or index == last:
            res = (index, self.threshold)
            self.avg_ticks = alpha*self.ticks + (1-alpha)*self.avg_ticks
            self.threshold = self.avg_ticks*abs(self.avg_imbalance)
            # reset
            self.ticks = 0
            self.cum_imbalance = 0
            # Note: I found that it is better idea to reset
            # avg_imbalance with zero than with the last imbalance
            # zero is much closer to the expected value for the
            # long run.
            self.avg_imbalance = 0  # imbalance
        return res


class TickRunsEvents(TickTEvents):
    def __init__(self, imbalance, expected_window, expected_imbalance, alpha):
        self.cum_imbalance_u = 0
        self.cum_imbalance_d = 0
        super().__init__(imbalance, expected_window, expected_imbalance, alpha)

    def process_row(self, index, imbalance, last, alpha):
        # _imbalance = 0 if imbalance_isnull else imbalance
        self.ticks += 1
        self.avg_imbalance = alpha*imbalance + (1-alpha)*self.avg_imbalance
        self.cum_imbalance_u += imbalance if imbalance > 0 else 0
        self.cum_imbalance_d += imbalance if imbalance < 0 else 0
        stat = max(self.cum_imbalance_u, -self.cum_imbalance_d)
        self._statistics.append(stat)
        res = None
        if self._statistics[-1] >= self.threshold or index == last:
            res = (index, self.threshold)
            self.avg_ticks = alpha*self.ticks + (1-alpha)*self.avg_ticks
            # avg_imbalance = 2*p_b_up - 1
            p_b_up = (self.avg_imbalance + 1)/2
            self.threshold = self.avg_ticks*max(p_b_up, 1 - p_b_up)
            # reset
            self.ticks = 0
            self.cum_imbalance_d = 0
            self.cum_imbalance_u = 0
            # Note: I found that it is better idea to reset
            # avg_imbalance with zero than with the last imbalance
            # zero is much closer to the expected value for the
            # long run.
            self.avg_imbalance = 0  # imbalance
        return res


class DollarVolumeImbalanceEvents(DollarVolumeTEvents):
    def __init__(self, imbalance, volume, expected_window, expected_imbalance,
                 alpha):
        self.cum_imbalance = 0
        super().__init__(imbalance, volume, expected_window,
                         expected_imbalance, alpha)

    def process_row(self, index, imbalance, volume, last, alpha):
        # _imbalance = 0 if imbalance_isnull else imbalance
        self.ticks += 1
        _imbalance = imbalance*volume
        self.avg_imbalance = alpha*_imbalance + (1-alpha)*self.avg_imbalance
        self.cum_imbalance += _imbalance
        self._statistics.append(abs(self.cum_imbalance))
        res = None
        if self._statistics[-1] >= self.threshold or index == last:
            res = (index, self.threshold)
            self.avg_ticks = alpha*self.ticks + (1-alpha)*self.avg_ticks
            self.threshold = self.avg_ticks*abs(self.avg_imbalance)
            # reset
            self.ticks = 0
            self.cum_imbalance = 0
            # Note: I found that it is better idea to reset
            # avg_imbalance with zero than with the last imbalance
            # zero is much closer to the expected value for the
            # long run.
            self.avg_imbalance = 0  # imbalance
        return res


class DollarVolumeRunsEvents(DollarVolumeTEvents):
    def __init__(self, imbalance, volume, expected_window, expected_imbalance,
                 expected_volume_up, expected_volume_down, alpha):
        self.cum_imbalance_u = 0
        self.cum_imbalance_d = 0
        self.avg_volume_u = expected_volume_up
        self.avg_volume_d = expected_volume_down
        p_b_up = (self.avg_imbalance + 1)/2
        self.threshold = self.avg_ticks*max(p_b_up*self.avg_volume_u,
                                            (1 - p_b_up)*self.avg_volume_d)
        self.volume_u = []
        self.volume_d = []
        super().__init__(imbalance, volume, expected_window,
                         expected_imbalance, alpha)

    def process_row(self, index, imbalance, volume, last, alpha):
        self.ticks += 1
        self.avg_imbalance = alpha*imbalance + (1-alpha)*self.avg_imbalance
        if imbalance > 0:
            self.avg_volume_u = alpha*volume + (1-alpha)*self.avg_volume_u
            self.cum_imbalance_u += volume
            self.volume_u.append(volume)
        else:
            self.avg_volume_d = alpha*volume + (1-alpha)*self.avg_volume_d
            self.cum_imbalance_d += volume
            self.volume_d.append(volume)
        stat = max(self.cum_imbalance_u, self.cum_imbalance_d)
        self._statistics.append(stat)
        res = None
        if self._statistics[-1] >= self.threshold or index == last:
            res = (index, self.threshold)
            self.avg_ticks = alpha*self.ticks + (1-alpha)*self.avg_ticks
            # avg_imbalance = 2*p_b_up - 1
            p_b_up = (self.avg_imbalance + 1)/2
            self.threshold = self.avg_ticks*max(p_b_up*self.avg_volume_u,
                                                (1 - p_b_up)*self.avg_volume_d)
            # reset
            self.ticks = 0
            self.cum_imbalance_d = 0
            self.cum_imbalance_u = 0
            # Note: I found that it is better idea to reset
            # avg_imbalance with zero than with the last imbalance
            # zero is much closer to the expected value for the
            # long run.
            self.avg_imbalance = 0  # imbalance
            self.avg_volume_u = np.mean(self.volume_u)
            self.avg_volume_d = np.mean(self.volume_d)
            self.volume_u = []
            self.volume_d = []
        return res


def compute_events(df, n, grp_idx_func):
    df_aux = df.assign(
        grp_idx=grp_idx_func(df),
        grp=lambda row: row.grp_idx // n
    )
    s_aux = df_aux.grp.diff()
    return s_aux[s_aux == 1].index


def get_tick_events(df, n_ticks):
    return compute_events(df, n_ticks,
                          lambda df: list(range(df.shape[0])))


def get_volume_events(df, n_volume):
    return compute_events(df, n_volume,
                          lambda df: df['volume'].cumsum())


def get_dollar_events(df, n_dollar):
    return compute_events(df, n_dollar,
                          lambda df: (df['volume']*df['price']).cumsum())


def bars_sampling(df, tevents):
    aux = pd.Series(0, index=df.index)
    aux[tevents] = 1
    aux = np.cumsum(aux)
    return create_bars(df, aux, compute_bar)


def _bars_sampling(df, n, events_func):
    return bars_sampling(df, events_func(df, n))


def tick_bars_sampling(df, n_ticks):
    return bars_sampling(df, get_tick_events(df, n_ticks))


def volume_bars_sampling(df, n_volume):
    return bars_sampling(df, get_volume_events(df, n_volume))


def dollar_bars_sampling(df, n_dollar):
    return bars_sampling(df, get_dollar_events(df, n_dollar))


def time_bars_sampling(df, freq):
    return create_bars(df, pd.Grouper(key='time', freq=freq), compute_bar)

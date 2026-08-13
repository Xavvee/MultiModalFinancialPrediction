"""Pull hourly BTCUSDT candles from Binance.

yfinance only serves intraday history for roughly the last two years, so it
cannot cover 2017-2019. Binance's public klines endpoint reaches back to
August 2017 and needs no key.

These candles are what make the direction-of-causality test possible: daily
prices cannot distinguish sentiment that leads from sentiment that follows.
"""
import time
import datetime as dt
import requests
import pandas as pd

OUT = 'data/old_dataset/market/btc_hourly.parquet'
START = dt.datetime(2017, 8, 17)
END = dt.datetime(2019, 12, 1)


def run(symbol='BTCUSDT', interval='1h', out=OUT, start=START, end=END):
    cursor = int(start.timestamp() * 1000)
    stop = int(end.timestamp() * 1000)
    rows = []
    while cursor < stop:
        r = requests.get('https://api.binance.com/api/v3/klines',
                         params={'symbol': symbol, 'interval': interval,
                                 'startTime': cursor, 'limit': 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 3_600_000
        print(f'  {dt.datetime.fromtimestamp(batch[-1][0]/1000, dt.UTC)}  '
              f'({len(rows):,} candles)', flush=True)
        time.sleep(0.25)          # stay well inside the public rate limit

    df = pd.DataFrame(rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'trades', 'tbav', 'tqav', 'ignore'])
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df = df[['ts', 'close', 'volume']].drop_duplicates('ts').sort_values('ts')

    df.to_parquet(out, index=False)
    gaps = df['ts'].diff().dt.total_seconds().div(3600)
    print(f'\nsaved {len(df):,} candles -> {out}')
    print(f'range: {df["ts"].min()} -> {df["ts"].max()}   missing hours: {int((gaps > 1).sum())}')


if __name__ == '__main__':
    run()

"""Shared loaders and statistics used across the analysis modules.

Every number reported in the thesis comes from this package. Keeping the data
access and the inference helpers in one place means a change to, say, the
minimum-tweets threshold propagates everywhere instead of drifting between
scripts.
"""
import numpy as np
import pandas as pd
from scipy import stats

# Paths. Datasets are gitignored; these are where the pipelines write them.
OLD_TWEETS = 'data/old_dataset/processed/per_tweet.parquet'
OLD_MARKET = 'data/old_dataset/market/market_features.csv'
OLD_INTRADAY = 'data/old_dataset/processed/intraday_meta.parquet'
OLD_HOURLY = 'data/old_dataset/market/btc_hourly.parquet'
NEW_TWEETS = 'data/new_dataset/processed/per_tweet.parquet'
NEW_MARKET = 'data/new_dataset/market/market_features_2021_23.csv'

# A daily sentiment average built from a handful of tweets is mostly sampling
# noise - see analysis/verification.py, which measures how the correlation
# decays as the daily sample shrinks.
MIN_TWEETS_PER_DAY = 200
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_market(path, next_col=True):
    m = pd.read_csv(path)
    m['date'] = pd.to_datetime(m['date'])
    m = m.sort_values('date').set_index('date')
    m['absret'] = m['daily_return'].abs()
    if next_col:
        m['next_return'] = m['daily_return'].shift(-1)
        m['next_absret'] = m['absret'].shift(-1)
        m['prev_return'] = m['daily_return'].shift(1)
    return m


def daily_sentiment(tweets_path, min_tweets=MIN_TWEETS_PER_DAY, extra=None):
    """Collapse per-tweet scores into one row per day."""
    df = pd.read_parquet(tweets_path)
    df = df[df['date'].notna() & df['finbert'].notna()]
    agg = {'finbert': ('finbert', 'mean'), 'roberta': ('roberta', 'mean'),
           'disp': ('finbert', 'std'), 'n': ('finbert', 'size')}
    if extra:
        agg.update(extra)
    daily = df.groupby('date').agg(**agg)
    return daily[daily['n'] >= min_tweets]


def joined(tweets_path, market_path, min_tweets=MIN_TWEETS_PER_DAY):
    """Daily sentiment aligned with market features - the workhorse frame."""
    return daily_sentiment(tweets_path, min_tweets).join(
        load_market(market_path), how='inner').dropna(subset=['daily_return'])


def load_hourly_btc():
    """Hourly sentiment + price, the frame behind the direction-of-causality test."""
    meta = pd.read_parquet(OLD_INTRADAY)
    sent = pd.read_parquet(OLD_TWEETS)
    if len(meta) != len(sent):
        raise RuntimeError(
            f'intraday_meta ({len(meta):,}) and per_tweet ({len(sent):,}) row counts '
            'differ - the positional join would be invalid. Re-run tools/etl_intraday.py.')
    meta['finbert'] = sent['finbert'].values
    meta['roberta'] = sent['roberta'].values

    px = pd.read_parquet(OLD_HOURLY)
    px['ts'] = pd.to_datetime(px['ts']).dt.floor('h')
    px = px.drop_duplicates('ts').set_index('ts').sort_index()
    px['ret'] = px['close'].pct_change()
    px['absret'] = px['ret'].abs()
    return meta, px


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def ci(r, n, alpha=ALPHA):
    """Confidence interval for a correlation, via the Fisher transform."""
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    k = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - k * se)), float(np.tanh(z + k * se))


def mde(n, alpha=ALPHA, power=0.80):
    """Smallest correlation detectable at the given power - the number that turns
    'we found nothing' into 'we exclude anything above this'."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float(np.tanh((z_a + z_b) / np.sqrt(n - 3)))


def report_corr(x, y, label, alpha=ALPHA):
    """Correlation with its interval and the effect it could have detected."""
    mask = pd.notna(x) & pd.notna(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        print(f'  {label:44s} (insufficient data)')
        return None
    r, p = stats.pearsonr(x, y)
    lo, hi = ci(r, len(x))
    flag = '  SIGNIFICANT' if p < alpha else ''
    print(f'  {label:44s} r={r:+.4f} [{lo:+.3f},{hi:+.3f}] p={p:.4f} '
          f'n={len(x):,} (MDE {mde(len(x)):.3f}){flag}')
    return r, p


def hac_slope(frame, target, key, controls, dummies=None, lags=24):
    """OLS slope on `key` with Newey-West errors.

    Autocorrelated series make naive standard errors far too confident; every
    time-series regression in this project goes through here or through a
    cluster-robust equivalent.
    """
    import statsmodels.api as sm
    X = frame[[key] + list(controls)].copy()
    if dummies is not None:
        X = X.join(dummies)
    X = sm.add_constant(X)
    y = frame[target]
    ok = X.notna().all(axis=1) & y.notna()
    m = sm.OLS(y[ok], X[ok]).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return m.params[key], m.bse[key], m.pvalues[key], int(m.nobs)


def hour_dummies(index):
    return pd.get_dummies(index.hour, prefix='h', drop_first=True
                          ).set_index(index).astype(float)

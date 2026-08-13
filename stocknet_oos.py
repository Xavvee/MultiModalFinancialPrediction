import glob
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

"""Out-of-sample validation of the overnight-sentiment result.

The finding survives every in-sample control, but split-half by time is a weak
form of validation - the same companies appear on both sides. The serious
question is whether the relationship generalises to firms the model never saw.

Three designs, in increasing strictness:

  A. COMPANY HOLDOUT   5-fold cross-validation over companies. Coefficients are
                       estimated on four fifths of the firms and used to predict
                       gaps for the remaining fifth. Company fixed effects cannot
                       transfer to unseen firms, so predictions use the pooled
                       intercept only.

  B. TIME HOLDOUT      fit on the earlier period, predict the later one.

  C. BOTH AT ONCE      unseen companies AND a later period - nothing about the
                       test rows was available when the model was fitted.

The comparison that matters is not the t-statistic but whether adding sentiment
improves prediction OUT OF SAMPLE against the identical model without it. An
in-sample coefficient can be a fluke; beating a control model on unseen firms
is much harder to fake.
"""

TWEETS = 'data/stocknet/processed/per_tweet.parquet'
PRICES = 'data/stocknet/price/raw'
CLOSE_UTC, OPEN_UTC = 21, 13.5
MIN_TWEETS = 3
CONTROLS = ['prev_return', 'prev_gap', 'prev_intraday', 'log_n']
N_FOLDS = 5
SEED = 17


def load_prices():
    frames = []
    for path in sorted(glob.glob(os.path.join(PRICES, '*.csv'))):
        t = os.path.splitext(os.path.basename(path))[0]
        d = pd.read_csv(path)
        d['date'] = pd.to_datetime(d['Date'])
        d['ticker'] = t
        frames.append(d[['ticker', 'date', 'Open', 'Close']])
    px = pd.concat(frames, ignore_index=True).sort_values(['ticker', 'date'])
    g = px.groupby('ticker')
    px['prev_close'] = g['Close'].shift(1)
    px['gap'] = px['Open'] / px['prev_close'] - 1
    px['intraday'] = px['Close'] / px['Open'] - 1
    px['daily_return'] = px['Close'] / px['prev_close'] - 1
    px['prev_return'] = g['daily_return'].shift(1)
    px['prev_gap'] = g['gap'].shift(1)
    px['prev_intraday'] = g['intraday'].shift(1)
    return px


def build():
    tw = pd.read_parquet(TWEETS)
    tw = tw[tw['finbert'].notna()].copy()
    tw['ts'] = pd.to_datetime(tw['ts'])
    hour = tw['ts'].dt.hour + tw['ts'].dt.minute / 60.0
    day = tw['ts'].dt.floor('D')
    ac = hour >= CLOSE_UTC
    tw['session'] = day.where(~ac, day + pd.Timedelta(days=1))
    tw = tw[ac | (hour < OPEN_UTC)]

    agg = (tw.groupby(['ticker', 'session'])
             .agg(sent=('finbert', 'mean'), n=('finbert', 'size')).reset_index()
             .rename(columns={'session': 'date'}))
    d = agg.merge(load_prices(), on=['ticker', 'date'], how='inner')
    d = d[d['n'] >= MIN_TWEETS].copy()
    d['log_n'] = np.log(d['n'])
    d['ticker'] = d['ticker'].astype(str)
    d = d.dropna(subset=['gap', 'sent'] + CONTROLS).reset_index(drop=True)
    # Standardise using training statistics only where it matters; here the
    # scale is global and does not depend on the outcome, so it cannot leak.
    d['sent_z'] = (d['sent'] - d['sent'].mean()) / d['sent'].std()
    return d


def fit_predict(train, test, use_sentiment):
    cols = CONTROLS + (['sent_z'] if use_sentiment else [])
    Xtr = sm.add_constant(train[cols], has_constant='add')
    m = sm.OLS(train['gap'], Xtr).fit()
    Xte = sm.add_constant(test[cols], has_constant='add')
    Xte = Xte[Xtr.columns]
    return m.predict(Xte).values, m


def evaluate(pred, actual, label):
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    if len(pred) < 50 or np.std(pred) == 0:
        return None
    r, p = stats.pearsonr(pred, actual)
    da = (np.sign(pred) == np.sign(actual)).mean() * 100
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    return {'label': label, 'n': len(pred), 'r': r, 'p': p, 'da': da, 'rmse': rmse}


def report(rows):
    for r in rows:
        if r is None:
            continue
        print(f'  {r["label"]:34s} n={r["n"]:5,}  r={r["r"]:+.4f} (p={r["p"]:.4f})  '
              f'DA={r["da"]:.2f}%  RMSE={r["rmse"]*100:.4f}%')


def design_a(d):
    """Company holdout: predict firms the model never saw."""
    print(f'\n=== A. COMPANY HOLDOUT ({N_FOLDS}-fold over {d["ticker"].nunique()} firms) ===')
    rng = np.random.default_rng(SEED)
    firms = np.array(sorted(d['ticker'].unique()))
    rng.shuffle(firms)
    folds = np.array_split(firms, N_FOLDS)

    oof = {True: [], False: []}
    actual = []
    for k, held in enumerate(folds):
        test = d[d['ticker'].isin(held)]
        train = d[~d['ticker'].isin(held)]
        if len(test) < 50:
            continue
        for use in (False, True):
            pred, _ = fit_predict(train, test, use)
            oof[use].append(pred)
        actual.append(test['gap'].values)

    actual = np.concatenate(actual)
    rows = [evaluate(np.concatenate(oof[False]), actual, 'controls only'),
            evaluate(np.concatenate(oof[True]), actual, 'controls + sentiment')]
    report(rows)
    gain(rows)
    return rows


def design_b(d):
    """Time holdout: fit early, predict late."""
    print('\n=== B. TIME HOLDOUT (fit on the earlier half) ===')
    mid = d['date'].quantile(0.5)
    train, test = d[d['date'] <= mid], d[d['date'] > mid]
    print(f'  train to {pd.Timestamp(mid).date()}: {len(train):,} | test: {len(test):,}')
    rows = []
    for use, lbl in [(False, 'controls only'), (True, 'controls + sentiment')]:
        pred, _ = fit_predict(train, test, use)
        rows.append(evaluate(pred, test['gap'].values, lbl))
    report(rows)
    gain(rows)
    return rows


def design_c(d):
    """Unseen companies AND a later period."""
    print('\n=== C. BOTH: unseen firms in a later period ===')
    rng = np.random.default_rng(SEED + 1)
    firms = np.array(sorted(d['ticker'].unique()))
    rng.shuffle(firms)
    held = set(firms[:len(firms) // 5])
    mid = d['date'].quantile(0.5)
    train = d[(~d['ticker'].isin(held)) & (d['date'] <= mid)]
    test = d[(d['ticker'].isin(held)) & (d['date'] > mid)]
    print(f'  train {len(train):,} rows ({len(firms)-len(held)} firms, early) | '
          f'test {len(test):,} rows ({len(held)} firms, late)')
    rows = []
    for use, lbl in [(False, 'controls only'), (True, 'controls + sentiment')]:
        pred, _ = fit_predict(train, test, use)
        rows.append(evaluate(pred, test['gap'].values, lbl))
    report(rows)
    gain(rows)
    return rows


def gain(rows):
    a, b = rows[0], rows[1]
    if a is None or b is None:
        return
    print(f'  -> sentiment changes out-of-sample correlation by '
          f'{b["r"]-a["r"]:+.4f}, directional accuracy by {b["da"]-a["da"]:+.2f} p.p., '
          f'RMSE by {(b["rmse"]-a["rmse"])*100:+.5f} p.p.')


def main():
    d = build()
    print(f'{len(d):,} company-sessions | {d["ticker"].nunique()} firms | '
          f'{d["date"].min().date()} -> {d["date"].max().date()}')
    print('Baseline for directional accuracy: '
          f'{max((d["gap"] > 0).mean(), (d["gap"] <= 0).mean())*100:.2f}% '
          '(always calling the more common direction)')
    design_a(d)
    design_b(d)
    design_c(d)


if __name__ == '__main__':
    main()

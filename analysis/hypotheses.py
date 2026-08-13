"""Alternative explanations, each tested and each refuted.

Ruling these out is what makes the main null credible: the absence of a
predictive signal is not down to the wrong construct, the wrong horizon, a
noisy corpus, or the wrong dependent variable. Every one of those was checked.

The important one is DISPERSION, because it nearly passed. It survived
Newey-West standard errors AND a moving-block bootstrap, and was refuted only by
the placebo - a reminder that robust inference alone does not establish
direction on time-series data.

Reproduces: journal sections 05 (table) and 06.
"""
import numpy as np
import pandas as pd
from scipy import stats

from analysis.common import (OLD_TWEETS, OLD_MARKET, load_hourly_btc, load_market,
                             joined, hac_slope, hour_dummies, report_corr)

MIN_TWEETS_PER_HOUR = 30


def _hourly_frame():
    meta, px = load_hourly_btc()
    meta['hour'] = meta['ts'].dt.floor('h')
    g = meta.groupby('hour')
    h = pd.DataFrame({'finbert': g['finbert'].mean(), 'roberta': g['roberta'].mean(),
                      'disp': g['finbert'].std(), 'n': g.size()})
    h = h[h['n'] >= MIN_TWEETS_PER_HOUR]

    d = h.join(px[['ret', 'absret']], how='inner').sort_index()
    d['log_n'] = np.log(d['n'])
    d['next_ret'] = px['ret'].reindex(d.index + pd.Timedelta(hours=1)).values
    d['next_absret'] = px['absret'].reindex(d.index + pd.Timedelta(hours=1)).values
    d['prev_absret'] = px['absret'].reindex(d.index - pd.Timedelta(hours=1)).values
    for k in [1, 2, 3, 6, 12, 24]:
        d[f'lag{k}'] = px['absret'].reindex(d.index - pd.Timedelta(hours=k)).values
    d['disp_z'] = (d['disp'] - d['disp'].mean()) / d['disp'].std()
    return d.dropna(subset=['disp_z', 'absret', 'next_absret', 'log_n'])


def dispersion():
    """Disagreement -> volatility. Theory (Miller 1977; Hong & Stein 2003) says
    dispersion of opinion drives volume and volatility, and volatility - unlike
    direction - really is forecastable. The test therefore asks whether
    disagreement adds anything ON TOP of what past volatility already explains."""
    d = _hourly_frame()
    fe = hour_dummies(d.index)
    print(f'=== Disagreement -> volatility  (n = {len(d):,} hours) ===')

    rich = ['absret', 'lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'lag24', 'log_n']
    # The placebo target is the previous hour, so lag1 IS that target and must be
    # dropped from its control set - leaving it in puts the dependent variable on
    # both sides and produces a meaningless coefficient.
    placebo_ctrl = [c for c in rich if c != 'lag1']

    for label, target, ctrl in [
            ('FORWARD  dispersion -> next hour', 'next_absret', rich),
            ('PLACEBO  dispersion -> previous hour', 'prev_absret', placebo_ctrl)]:
        c, se, p, n = hac_slope(d, target, 'disp_z', ctrl, fe)
        print(f'  {label:42s} coef={c:+.7f} t={c/se:+6.2f} p={p:.4f} n={n:,}')

    print('\n  The backward link is the stronger of the two: dispersion accompanies')
    print('  volatility rather than anticipating it. Note that this result passed')
    print('  Newey-West errors and a block bootstrap before the placebo caught it.')


def residual_and_horizons():
    """Two cheap alternatives: strip out the part of sentiment that today's move
    explains, and look further ahead than one day."""
    j = joined(OLD_TWEETS, OLD_MARKET)
    print(f'\n=== Sentiment residual and longer horizons  (n = {len(j):,} days) ===')

    for model in ['finbert', 'roberta']:
        x, y = j[model].values, j['daily_return'].values
        beta, alpha = np.polyfit(y, x, 1)
        resid = x - (alpha + beta * y)
        fwd = j['next_return'].values
        ok = ~np.isnan(fwd)
        report_corr(x[ok], fwd[ok], f'{model} raw level -> next day')
        report_corr(resid[ok], fwd[ok], f'{model} RESIDUAL -> next day')

    close = j['Close']
    for h in [2, 3, 5, 10, 20]:
        fwd = (close.shift(-h) / close - 1).values
        ok = ~np.isnan(fwd)
        report_corr(j['finbert'].values[ok], fwd[ok], f'finbert -> {h}-day forward return')


def attention_and_bots():
    """Does raw attention forecast volatility, and does removing automated
    accounts sharpen anything?"""
    j = joined(OLD_TWEETS, OLD_MARKET)
    print(f'\n=== Attention -> volatility  (n = {len(j):,} days) ===')
    report_corr(np.log(j['n']), j['next_absret'], 'log(tweet volume) -> next |return|')
    report_corr(j['absret'], j['next_absret'], '[reference] today |return| -> next')

    df = pd.read_parquet(OLD_TWEETS)
    df = df[df['date'].notna() & df['finbert'].notna()].copy()
    for c in ['likes', 'retweets', 'replies']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['reach'] = df['likes'] + 20 * df['retweets'] + 15 * df['replies']

    acct = df.groupby('user', observed=True).agg(
        tweets=('finbert', 'size'), days=('date', 'nunique'),
        sd=('finbert', 'std'), zero=('reach', lambda s: (s == 0).mean()))
    acct['per_day'] = acct['tweets'] / acct['days']
    active = acct[acct['tweets'] >= 20]
    # Two of three behavioural flags: posts many times a day, produces
    # near-constant sentiment (a template), draws no audience at all.
    score = ((active['per_day'] >= 10).astype(int)
             + (active['sd'] <= 0.05).astype(int)
             + ((active['zero'] >= 0.98) & (active['tweets'] >= 50)).astype(int))
    bots = active.index[score >= 2]
    is_bot = df['user'].isin(bots)
    print(f'\n=== Automated accounts  ({len(bots):,} flagged, '
          f'{is_bot.mean()*100:.1f}% of tweets) ===')

    market = load_market(OLD_MARKET)
    clean = df[~is_bot]
    # Compare on IDENTICAL days: filtering changes which days clear the volume
    # threshold, and denser days correlate more strongly regardless of bots.
    counts = clean.groupby('date').size()
    days = counts[counts >= 200].index
    for label, src in [('all tweets', df), ('bots removed', clean)]:
        daily = src[src['date'].isin(days)].groupby('date')[['finbert']].mean()
        jj = daily.join(market, how='inner').dropna(subset=['daily_return'])
        report_corr(jj['finbert'], jj['daily_return'], f'{label}, same-day')
    print('  -> removing them does not help; the earlier apparent gain was a')
    print('     day-selection artifact, which is why the days are matched here.')


def feedback():
    """Does negative sentiment amplify a fall? Regime splits so far used
    volatility and attention, never the SIGN of the move."""
    d = _hourly_frame()
    d['down'] = (d['ret'] < 0).astype(float)
    for m in ['finbert', 'roberta']:
        d[f'{m}_z'] = (d[m] - d[m].mean()) / d[m].std()
        d[f'{m}_down'] = d[f'{m}_z'] * d['down']
    d = d.dropna(subset=['next_ret', 'ret'])
    fe = hour_dummies(d.index)
    ctrl = ['ret', 'log_n', 'down']
    print(f'\n=== Feedback: does sentiment weigh more while price falls?  '
          f'(n = {len(d):,}) ===')
    for m in ['finbert', 'roberta']:
        import statsmodels.api as sm
        X = sm.add_constant(d[[f'{m}_z', f'{m}_down'] + ctrl].join(fe))
        y = d['next_ret']
        ok = X.notna().all(axis=1) & y.notna()
        fit = sm.OLS(y[ok], X[ok]).fit(cov_type='HAC', cov_kwds={'maxlags': 24})
        for k in [f'{m}_z', f'{m}_down']:
            c, se, p = fit.params[k], fit.bse[k], fit.pvalues[k]
            print(f'  {k:20s} coef={c:+.6f} t={c/se:+5.2f} p={p:.3f}')
    print('  -> no main effect and no interaction: falls are not self-reinforcing here.')


def model_comparison():
    """Report the two language models separately - averaging them would hide
    that they agree only moderately and that one is clearly the better tool."""
    d = _hourly_frame()
    print('\n=== FinBERT vs RoBERTa ===')
    print(f'  agreement between the two series: r = {d["finbert"].corr(d["roberta"]):+.3f}')
    for m in ['finbert', 'roberta']:
        print(f'  {m:8s} mean {d[m].mean():+.4f}  sd {d[m].std():.4f}  '
              f'vs same-hour return r={d[m].corr(d["ret"]):+.4f}  '
              f'vs next-hour r={d[m].corr(d["next_ret"]):+.4f}')
    print('  -> FinBERT tracks price better despite being trained on financial')
    print('     prose rather than social posts; RoBERTa skews positive.')


def run():
    dispersion()
    residual_and_horizons()
    attention_and_bots()
    feedback()
    model_comparison()


if __name__ == '__main__':
    run()

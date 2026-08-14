import numpy as np
import pandas as pd
import statsmodels.api as sm

from reddit_analysis import daily_frame

"""Does Reddit attention forecast volatility, or merely accompany it?

The daily screen turned up one significant result: post volume correlates with
the NEXT day's absolute return (r = 0.145, p = 0.006). That is exactly the shape
of the two findings this project has already had to discard, so it gets the same
treatment before it is believed.

Today's volatility already predicts tomorrow's (r = 0.136), and busy days are
busy precisely because the price moved. So the question is whether attention
adds anything ON TOP of the volatility history - and whether it also "predicts"
volatility that has already happened, which would mark it as contemporaneous
rather than forward-looking.

Note the same trap that caught the earlier analysis: the placebo target is
yesterday's volatility, so yesterday's volatility must be REMOVED from the
placebo's control set, or the dependent variable sits on both sides.
"""

HAC_LAGS = 7


def frame():
    _, d = daily_frame()
    d = d.sort_index().copy()
    d['log_n'] = np.log(d['n'])
    for k in [1, 2, 3, 5]:
        d[f'lag{k}'] = d['absret'].shift(k)
    d['prev_absret'] = d['lag1']
    d['log_n_z'] = (d['log_n'] - d['log_n'].mean()) / d['log_n'].std()
    d['upvote_z'] = (d['upvote'] - d['upvote'].mean()) / d['upvote'].std()
    return d.dropna(subset=['next_absret', 'absret', 'lag1', 'lag2', 'lag3', 'lag5'])


def fit(d, target, key, controls):
    X = sm.add_constant(d[[key] + list(controls)])
    y = d[target]
    ok = X.notna().all(axis=1) & y.notna()
    m = sm.OLS(y[ok], X[ok]).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
    c, se, p = m.params[key], m.bse[key], m.pvalues[key]
    return c, c / se, p, int(m.nobs)


def run():
    d = frame()
    print(f'n = {len(d)} days\n')

    rich = ['absret', 'lag1', 'lag2', 'lag3', 'lag5']
    placebo_ctrl = [c for c in rich if c != 'lag1']   # lag1 IS the placebo target

    for key, name in [('log_n_z', 'post volume'), ('upvote_z', 'upvote ratio')]:
        print(f'=== {name} ===')
        for label, ctrl in [('no controls', []),
                            ('+ today volatility', ['absret']),
                            ('+ volatility history (5 lags)', rich)]:
            c, t, p, n = fit(d, 'next_absret', key, ctrl)
            tag = 'SIGNIFICANT' if p < 0.05 else 'dead'
            print(f'  forward  {label:32s} coef={c:+.6f} t={t:+5.2f} p={p:.4f} {tag}')
        c, t, p, n = fit(d, 'prev_absret', key, placebo_ctrl)
        tag = 'SIGNIFICANT' if p < 0.05 else 'dead'
        print(f'  PLACEBO  {"-> yesterday volatility":32s} coef={c:+.6f} t={t:+5.2f} p={p:.4f} {tag}\n')

    print('=== Reading ===')
    _, t_f, p_f, _ = fit(d, 'next_absret', 'log_n_z', rich)
    _, t_p, p_p, _ = fit(d, 'prev_absret', 'log_n_z', placebo_ctrl)
    if p_f < 0.05 and p_p >= 0.05:
        print('  Volume survives the full control set and the placebo is dead:')
        print('  attention carries genuine forward information about volatility.')
    elif p_f >= 0.05:
        print('  Volume dies once volatility history is controlled for - the raw')
        print('  correlation was past volatility showing through.')
    else:
        print('  Both alive: attention tracks volatility in both directions, which')
        print('  is the signature of a contemporaneous relationship, not a forecast.')
        print(f'  forward t={t_f:+.2f} vs backward t={t_p:+.2f} - compare magnitudes,')
        print('  not just significance.')


if __name__ == '__main__':
    run()

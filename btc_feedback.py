import numpy as np
import pandas as pd
import statsmodels.api as sm

"""Does negative sentiment amplify a fall? And do the two models differ?

FEEDBACK. Behavioural finance predicts asymmetry: fear moves faster than greed,
and a falling price plus negative commentary is the textbook panic loop. Nothing
tested so far could detect it - regime splits used volatility and attention, never
the SIGN of the move. Tested here as an interaction: does sentiment carry
different predictive weight while the market is already falling?

MODEL SPLIT. FinBERT is trained on financial prose, RoBERTa on tweets. They
disagree often; reporting them separately shows whether any conclusion depends
on the choice, which a single averaged number would hide.

Every forward test is paired with its backward placebo, since that is what
exposed the two false positives earlier in this project.
"""

MIN_TWEETS, HAC = 30, 24

meta = pd.read_parquet('data/old_dataset/processed/intraday_meta.parquet')
sent = pd.read_parquet('data/old_dataset/processed/per_tweet.parquet')
assert len(meta) == len(sent)
meta['finbert'] = sent['finbert'].values
meta['roberta'] = sent['roberta'].values
del sent

px = pd.read_parquet('data/old_dataset/market/btc_hourly.parquet')
px['ts'] = pd.to_datetime(px['ts']).dt.floor('h')
px = px.drop_duplicates('ts').set_index('ts').sort_index()
px['ret'] = px['close'].pct_change()
px['absret'] = px['ret'].abs()

meta['hour'] = meta['ts'].dt.floor('h')
g = meta.groupby('hour')
h = pd.DataFrame({'finbert': g['finbert'].mean(), 'roberta': g['roberta'].mean(),
                  'n': g.size()})
h = h[h['n'] >= MIN_TWEETS]

d = h.join(px[['ret', 'absret']], how='inner').sort_index()
d['log_n'] = np.log(d['n'])
d['next_ret'] = px['ret'].reindex(d.index + pd.Timedelta(hours=1)).values
d['prev_ret'] = px['ret'].reindex(d.index - pd.Timedelta(hours=1)).values
for k in [1, 2, 3, 6]:
    d[f'lag{k}'] = px['ret'].reindex(d.index - pd.Timedelta(hours=k)).values
d['down'] = (d['ret'] < 0).astype(float)
for m in ['finbert', 'roberta']:
    d[f'{m}_z'] = (d[m] - d[m].mean()) / d[m].std()
    d[f'{m}_down'] = d[f'{m}_z'] * d['down']
d = d.dropna(subset=['next_ret', 'prev_ret', 'ret', 'lag1', 'lag2', 'lag3', 'lag6'])
hour_fe = pd.get_dummies(d.index.hour, prefix='h', drop_first=True).set_index(d.index).astype(float)
CTRL = ['ret', 'lag1', 'lag2', 'lag3', 'lag6', 'log_n', 'down']
# lag1 IS the previous hour's return, which is also the placebo's target. Leaving
# it in the control set would put the dependent variable on both sides of the
# regression and produce a meaningless coefficient, so the placebo drops it.
PLACEBO_CTRL = [c for c in CTRL if c != 'lag1']

print(f'n = {len(d):,} hours   |   down-hours: {int(d["down"].sum()):,} '
      f'({d["down"].mean()*100:.1f}%)\n')


def run(label, target, keys, extra=(), controls=None):
    ctrl = CTRL if controls is None else controls
    X = d[list(keys) + list(ctrl) + list(extra)].join(hour_fe)
    X = sm.add_constant(X)
    y = d[target]
    ok = X.notna().all(axis=1) & y.notna()
    m = sm.OLS(y[ok], X[ok]).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})
    parts = []
    for k in keys:
        c, se, p = m.params[k], m.bse[k], m.pvalues[k]
        parts.append(f'{k}: coef={c:+.6f} t={c/se:+5.2f} p={p:.3f}'
                     f'{" *" if p < 0.05 else "  "}')
    print(f'  {label:38s} ' + ' | '.join(parts))
    return {k: m.pvalues[k] for k in keys}


print('=== FEEDBACK: does sentiment matter more while price is falling? ===')
for m in ['finbert', 'roberta']:
    print(f'  -- {m} --')
    run('forward: next-hour return', 'next_ret', [f'{m}_z', f'{m}_down'])
    run('PLACEBO: previous-hour return', 'prev_ret', [f'{m}_z', f'{m}_down'],
        controls=PLACEBO_CTRL)

print('\n=== Same question restricted to sharp falls (worst 20% of hours) ===')
cut = d['ret'].quantile(0.20)
sub = d[d['ret'] <= cut]
print(f'  hours with return <= {cut*100:.2f}%: {len(sub):,}')
for m in ['finbert', 'roberta']:
    X = sm.add_constant(sub[[f'{m}_z', 'ret', 'lag1', 'log_n']])
    fit = sm.OLS(sub['next_ret'], X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})
    c, se, p = fit.params[f'{m}_z'], fit.bse[f'{m}_z'], fit.pvalues[f'{m}_z']
    print(f'  {m:8s} sentiment -> next return after a sharp fall: '
          f'coef={c:+.6f} t={c/se:+5.2f} p={p:.3f}{"  SIGNIFICANT" if p < 0.05 else ""}')

print('\n=== MODEL COMPARISON: how much do the two actually agree? ===')
print(f'  correlation between the two daily-hourly series: '
      f'{d["finbert"].corr(d["roberta"]):+.3f}')
print(f'  finbert: mean {d["finbert"].mean():+.4f}  sd {d["finbert"].std():.4f}')
print(f'  roberta: mean {d["roberta"].mean():+.4f}  sd {d["roberta"].std():.4f}')
for m in ['finbert', 'roberta']:
    r = d[m].corr(d['ret'])
    r_next = d[m].corr(d['next_ret'])
    print(f'  {m:8s} vs same-hour return r={r:+.4f}   vs next-hour r={r_next:+.4f}')

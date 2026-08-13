"""What could this study have detected, and what do its nulls exclude?

A null result is worth exactly as much as the power behind it. "We found
nothing" is weak. "We would have caught anything above r = 0.09, and measured
0.012, so we exclude everything above 0.076" is a quantitative statement a
reviewer can check.

The analytic formula is verified by simulation here rather than trusted, and
each reported null is converted into a confidence interval.

Reproduces: journal section 06.
"""
import numpy as np
from scipy import stats

from analysis.common import ci, mde, ALPHA

# Observed nulls worth converting into exclusions: (label, n, r)
OBSERVED = [
    ('2016-19 sentiment -> next-day return', 942, 0.0117),
    ('2016-19 roberta -> next-day return', 942, 0.0534),
    ('2016-19 volume -> next-day volatility', 942, -0.0021),
    ('2016-19 top-1% reach -> next-day', 301, 0.0332),
    ('2021-23 retail -> next-day return', 222, 0.0532),
    ('2021-23 whales -> next-day return', 222, -0.1158),
]

# The same-day effect, for contrast - this is what a detectable effect looks like
SAME_DAY = [
    ('2016-19 same-day', 942, 0.2463),
    ('2021-23 same-day', 222, 0.4164),
    ('2016-19 dense days, same-day', 254, 0.3601),
]


def simulate_power(n, r_true, reps=4000, seed=0):
    """Empirical rejection rate - confirms the analytic MDE is not fiction."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        x = rng.standard_normal(n)
        y = r_true * x + np.sqrt(max(1 - r_true ** 2, 0)) * rng.standard_normal(n)
        if stats.pearsonr(x, y)[1] < ALPHA:
            hits += 1
    return hits / reps


def run():
    print('=== 1. Detectable effect by sample size (80% power) ===')
    print(f'{"sample":34s} {"n":>6s} {"MDE analytic":>13s} {"MDE simulated":>15s}')
    for label, n in [('2016-19, all usable days', 942),
                     ('2016-19, dense days only', 254),
                     ('2021-23, real days', 222),
                     ('hourly resolution', 5682)]:
        m = mde(n)
        print(f'{label:34s} {n:6d} {m:13.4f} {simulate_power(n, m)*100:14.0f}%')

    print('\n=== 2. What each null excludes ===')
    print(f'{"measurement":40s} {"n":>5s} {"r":>8s} {"95% CI":>20s}')
    for label, n, r in OBSERVED:
        lo, hi = ci(r, n)
        print(f'{label:40s} {n:5d} {r:+8.4f}   [{lo:+.3f}, {hi:+.3f}]')

    print('\n=== 3. For contrast, the effect that IS there ===')
    for label, n, r in SAME_DAY:
        lo, hi = ci(r, n)
        print(f'{label:40s} {n:5d} {r:+8.4f}   [{lo:+.3f}, {hi:+.3f}]')

    print('\n=== 4. The decisive comparison ===')
    for n in [222, 942]:
        p = simulate_power(n, 0.25)
        print(f'  power to detect r=0.25 (the same-day magnitude) at n={n:4d}: {p*100:.1f}%')
    print('  If sentiment predicted tomorrow as strongly as it reacts to today,')
    print('  we would have found it with near-certainty. We did not.')

    print('\n=== 5. Honest limitation ===')
    print(f'  The 2021-23 corpus has 222 usable days -> MDE {mde(222):.3f}.')
    print('  Its whale result (-0.116) cannot be separated from a real contrarian')
    print('  effect of that size; that corpus simply cannot settle the question.')
    print(f'  Per-account screens are weaker still: at 30 observations per account')
    print(f'  the detectable correlation is {mde(30):.2f}, so only very strong')
    print('  individual effects were ever in reach.')


if __name__ == '__main__':
    run()

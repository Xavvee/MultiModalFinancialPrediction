"""Builds one comparable benchmark table from the models' own predictions.

Why this exists: every model in this project derives its own test window.
Price-level models pull from yfinance, the GRU reads the sentiment pickle
(which ends earlier), and the LSTM variants each lose their own LOOK_BACK days
to windowing. The result was a table whose rows were scored on different days -
so "GRU 54.31% vs Majority 49.77%" compared 197 days against 213 and was not a
valid comparison at all.

Here every model hands back its dated predictions, those are reduced to one
directional call per date, and the calls are intersected down to the set of
dates EVERY model covers. Only then is accuracy computed.

Two accuracy columns are reported side by side, because the choice is not free:

  DA        - no-change predictions excluded from the denominator. Fair to a
              model that abstains, but the denominator becomes model-dependent.
  DA strict - no-change predictions counted as wrong. Identical denominator for
              every row, and it penalises a forecast that collapses toward a
              constant - which the permissive definition hides.

Coverage (share of days with a non-zero call) is what reconciles the two.
"""
import os
import numpy as np
import pandas as pd

from utils import (calculate_directional_accuracy, directional_accuracy_strict,
                   directional_coverage)


def _calls(result):
    """Reduces a model result to (dates, true_dir, pred_dir), one call per date.

    For returns models the call for date d uses that date's own sign. For
    price-level models the call is the implied change from the previous day, so
    it belongs to the LATER of the two dates - hence dates[1:].
    """
    dates = pd.DatetimeIndex(result['dates'])
    y_true = np.asarray(result['y_true'], dtype=float)
    y_pred = np.asarray(result['y_pred'], dtype=float)

    if result['is_stationary']:
        return dates, np.sign(y_true), np.sign(y_pred)
    return (dates[1:],
            np.sign(y_true[1:] - y_true[:-1]),
            np.sign(y_pred[1:] - y_true[:-1]))


def build(results, ticker, results_dir):
    """results: list of dicts returned by the models' run(); None entries skipped."""
    results = [r for r in results if r]
    if not results:
        return None

    per_model = {}
    for r in results:
        dates, td, pd_ = _calls(r)
        per_model[r['name']] = pd.DataFrame({'true': td, 'pred': pd_}, index=dates)

    common = None
    for frame in per_model.values():
        common = frame.index if common is None else common.intersection(frame.index)
    common = common.sort_values()

    if len(common) == 0:
        print(f"      !! No overlapping dates across models for {ticker} - table skipped.")
        return None

    rows = []
    for r in results:
        frame = per_model[r['name']].loc[common]
        true_dir, pred_dir = frame['true'].values, frame['pred'].values
        called = pred_dir != 0
        rows.append({
            'model': r['name'],
            'n': len(common),
            'from': str(common[0].date()),
            'to': str(common[-1].date()),
            'DA': (float((true_dir[called] == pred_dir[called]).mean() * 100)
                   if called.sum() else float('nan')),
            'DA_strict': float((true_dir == pred_dir).mean() * 100),
            'coverage': float(called.mean() * 100),
            'RMSE': r['rmse'],
        })

    table = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f'benchmark_{ticker}.csv')
    table.to_csv(out, index=False)

    print(f"\n      === BENCHMARK (common window: {len(common)} days, "
          f"{common[0].date()} .. {common[-1].date()}) ===")
    print(f"      {'model':22s} {'DA':>8} {'DA strict':>10} {'coverage':>9} {'RMSE':>10}")
    for _, row in table.iterrows():
        da = '   —' if np.isnan(row['DA']) else f"{row['DA']:7.2f}%"
        rmse = '        —' if row['RMSE'] is None else f"{row['RMSE']:9.5f}"
        print(f"      {row['model']:22s} {da:>8} {row['DA_strict']:9.2f}% "
              f"{row['coverage']:8.1f}% {rmse}")

    # Insisting on one window costs sample size - the intersection is only as
    # large as the most windowing-hungry model allows. State the resulting
    # resolution explicitly, so nobody reads a 5pp gap here as a real difference.
    margin = 1.96 * np.sqrt(0.25 / len(common)) * 100
    print(f"      ---")
    print(f"      At n={len(common)}, the 95% margin on a single accuracy near 50% is "
          f"+/-{margin:.1f} pp,")
    print(f"      so differences smaller than roughly {2*margin:.0f} pp between two rows "
          f"are not resolvable.")
    print(f"      saved -> {out}")
    return table

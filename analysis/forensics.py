"""Detecting a dataset whose dates have been tampered with.

A corpus published as "Bitcoin Tweets 2025-2026" turned out to be a copy of a
2021-2023 collection with the years rewritten. Every result computed on it was
meaningless by construction - sentiment from 2021 compared against prices from
2025. The suspicion started when sentiment failed to correlate with even the
SAME-DAY return, which no working sentiment measure does.

Two independent checks settled it, and both are reusable on any new corpus:

  ACCOUNT AGE     the newest account-creation date in the file must reach the
                  end of the claimed range. People join constantly; a corpus
                  running to 2026 in which nobody joined after January 2023 is
                  impossible. This needs no assumptions and takes seconds.

  EMBEDDED CLOCKS price bots paste the timestamp of their reading into the tweet
                  body ("Price: 4872644.0 (2021/02/11 08:51)"). Comparing that
                  against the row's own date recovers the true date directly.

Reproduces: journal section 01.
"""
import csv
import re
import collections
import datetime as dt
import pandas as pd

csv.field_size_limit(10 ** 7)
EMBEDDED_TS = re.compile(r'\((20[12][0-9])/([01][0-9])/([0-3][0-9])\s')


def account_age_gate(path, created_col='user_created', date_col='date',
                     sep=',', tolerance_days=120):
    """Newest account vs last post. The cheapest possible authenticity check."""
    newest = None
    last_post = None
    n = 0
    with open(path, encoding='utf-8', errors='ignore') as fh:
        reader = csv.reader(fh, delimiter=sep)
        header = next(reader)
        ci = header.index(created_col)
        di = header.index(date_col)
        for row in reader:
            if len(row) <= max(ci, di):
                continue
            for idx, keep in ((ci, 'created'), (di, 'post')):
                try:
                    ts = pd.Timestamp(row[idx][:10])
                except Exception:
                    continue
                if not (2006 <= ts.year <= 2030):
                    continue
                if keep == 'created':
                    n += 1
                    if newest is None or ts > newest:
                        newest = ts
                else:
                    if last_post is None or ts > last_post:
                        last_post = ts

    gap = (last_post - newest).days
    ok = gap <= tolerance_days
    print(f'  accounts read: {n:,}')
    print(f'  newest account : {newest.date()}')
    print(f'  last post      : {last_post.date()}')
    print(f'  gap            : {gap} days -> {"PASS" if ok else "FAIL - dates are not what they claim"}')
    return ok, newest, last_post


def recover_dates(path, date_col='date', text_col='text', sep=','):
    """Recover true dates from timestamps that bots paste into their own text."""
    pairs = collections.defaultdict(collections.Counter)
    rows = hits = 0
    with open(path, encoding='utf-8', errors='ignore') as fh:
        reader = csv.reader(fh, delimiter=sep)
        header = next(reader)
        di, ti = header.index(date_col), header.index(text_col)
        for row in reader:
            rows += 1
            if len(row) <= max(di, ti):
                continue
            m = EMBEDDED_TS.search(row[ti])
            if not m:
                continue
            try:
                emb = pd.Timestamp(f'{m.group(1)}-{m.group(2)}-{m.group(3)}')
                lab = pd.Timestamp(row[di][:10])
            except Exception:
                continue
            pairs[lab][emb] += 1
            hits += 1

    print(f'  rows scanned: {rows:,} | tweets carrying an embedded clock: {hits:,}')
    if not pairs:
        print('  no embedded timestamps found - this check does not apply here')
        return None

    recs = []
    for lab, counter in sorted(pairs.items()):
        emb, votes = counter.most_common(1)[0]
        recs.append({'label': lab, 'true': emb, 'offset_days': (lab - emb).days,
                     'agreement': votes / sum(counter.values())})
    mp = pd.DataFrame(recs)
    print(f'  labelled days covered: {len(mp)}')
    print(f'  offset range: {mp["offset_days"].min()} .. {mp["offset_days"].max()} days')
    print(f'  RECOVERED TRUE RANGE: {mp["true"].min().date()} -> {mp["true"].max().date()}')
    print(f'  LABELLED RANGE:       {mp["label"].min().date()} -> {mp["label"].max().date()}')
    if mp['offset_days'].nunique() > 1:
        print('  -> offset is NOT constant: the chronology has been scrambled, not just shifted')
    return mp


def run():
    print('=== Recorded findings ===')
    print('  "Bitcoin Tweets 2025-2026" (pokeash)')
    print('    accounts read: 4,660,872 | newest account 2023-01-09 | last post 2026-03-02')
    print('    gap: 1148 days -> FAIL')
    print('    embedded clocks: 3,189 tweets across all 199 labelled days')
    print('    recovered true range 2021-02-06 -> 2023-01-09, offsets 1095..1826 days')
    print('    -> a copy of kaushiksuresh147/bitcoin-tweets with the years rewritten;')
    print('       the same row appears in both, identical byte for byte apart from')
    print('       the year (2021-02-10 vs 2026-02-10)')
    print()
    print('  Genuine corpora, same checks:')
    print('    Bitcoin 2021-23  newest account 2023-01-09, last post 2023-01-09 -> PASS')
    print('    stocknet         newest account 2016-03-29, last post 2016-03-31 -> PASS')
    print()
    print('  To run against a new file:')
    print('    from analysis.forensics import account_age_gate, recover_dates')
    print('    account_age_gate("data/<corpus>.csv")')


if __name__ == '__main__':
    run()

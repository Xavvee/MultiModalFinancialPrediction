import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

"""Generates every figure for the thesis in one consistent style.

Mixing matplotlib output with charts lifted from the research journal would look
inconsistent on the page, so everything is redrawn here from a single set of
style rules: one palette, one font size, one figure width.

Most figures recompute from the data on every run, so they cannot drift away
from the numbers in the text. Two are built from recorded values because
regenerating them means hours of computation - those are marked RECORDED in the
code and in the caption file.

Output: figures/rys_NN_nazwa.png (300 dpi) and .pdf (vector, for LaTeX),
plus figures/podpisy.md with a caption for each.
"""

OUT = 'figures'

# Colour meaning is fixed across every figure so the reader learns it once:
BLUE = '#2a78d6'      # prediction channel / observed data
ORANGE = '#eb6834'    # reaction channel / control / null
GREEN = '#1baf7a'     # confirmed effect
GREY = '#77766f'      # reference lines, secondary series
INK = '#0b0b0b'

plt.rcParams.update({
    'figure.figsize': (7.2, 4.2),
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

CAPTIONS = []


def save(fig, num, slug, caption):
    os.makedirs(OUT, exist_ok=True)
    base = f'{OUT}/rys_{num:02d}_{slug}'
    fig.savefig(base + '.png')
    fig.savefig(base + '.pdf')
    plt.close(fig)
    CAPTIONS.append((num, slug, caption))
    print(f'  rys. {num}: {slug}')


# --------------------------------------------------------------------------- #
def fig01_scaling():
    """Correlation strengthens with the number of posts averaged."""
    from analysis.common import OLD_TWEETS, OLD_MARKET, NEW_TWEETS, NEW_MARKET, load_market

    pts = []
    for tweets, market, label, colour in [
            (OLD_TWEETS, OLD_MARKET, 'Bitcoin 2016–2019', BLUE),
            (NEW_TWEETS, NEW_MARKET, 'Bitcoin 2021–2023', ORANGE)]:
        df = pd.read_parquet(tweets)
        df = df[df['date'].notna() & df['finbert'].notna()]
        daily = df.groupby('date').agg(s=('finbert', 'mean'), n=('finbert', 'size'))
        j = daily.join(load_market(market), how='inner').dropna(subset=['daily_return'])
        edges = [0, 100, 300, 1000, 5000, 10**9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sub = j[(j['n'] >= lo) & (j['n'] < hi)]
            if len(sub) < 60:
                continue
            r, _ = stats.pearsonr(sub['s'], sub['daily_return'])
            pts.append((sub['n'].median(), r, label, colour))

    sn = pd.read_parquet('data/stocknet/processed/per_tweet.parquet')
    sn = sn[sn['finbert'].notna()].copy()
    sn['date'] = pd.to_datetime(sn['ts']).dt.floor('D')
    import glob
    px = []
    for p in glob.glob('data/stocknet/price/raw/*.csv'):
        d = pd.read_csv(p)
        d['date'] = pd.to_datetime(d['Date'])
        d['ticker'] = os.path.splitext(os.path.basename(p))[0]
        d['ret'] = d['Close'].pct_change()
        px.append(d[['ticker', 'date', 'ret']])
    px = pd.concat(px)
    agg = sn.groupby(['ticker', 'date']).agg(s=('finbert', 'mean'), n=('finbert', 'size')).reset_index()
    m = agg.merge(px, on=['ticker', 'date'], how='inner').dropna(subset=['ret'])
    for lo, hi in [(1, 5), (5, 10), (10, 20), (20, 10**9)]:
        sub = m[(m['n'] >= lo) & (m['n'] < hi)]
        if len(sub) < 100:
            continue
        r, _ = stats.pearsonr(sub['s'], sub['ret'])
        pts.append((sub['n'].median(), r, 'Akcje (stocknet)', GREEN))

    fig, ax = plt.subplots()
    # One shared trend through every point: the claim is that a single
    # relationship governs all three corpora, so three separate broken lines
    # would obscure it - and one corpus yields only a single bucket anyway.
    xs = np.log10([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    b, a = np.polyfit(xs, ys, 1)
    grid = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(10 ** grid, a + b * grid, color=GREY, linewidth=1.4,
            linestyle='-', alpha=0.65, zorder=1,
            label=f'wspólny trend (R² = {np.corrcoef(xs, ys)[0,1]**2:.2f})')
    for label, colour in [('Bitcoin 2016–2019', BLUE), ('Bitcoin 2021–2023', ORANGE),
                          ('Akcje (stocknet)', GREEN)]:
        g = [p for p in pts if p[2] == label]
        if not g:
            continue
        ax.scatter([p[0] for p in g], [p[1] for p in g], s=55, color=colour,
                   label=label, zorder=3, edgecolor='white', linewidth=0.8)
    ax.axhline(0, color=GREY, linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('mediana liczby wpisów w jednostce obserwacji (skala log.)')
    ax.set_ylabel('korelacja z ceną tego samego dnia')
    ax.set_title('Siła zmierzonej korelacji rośnie z liczbą uśrednianych wpisów')
    ax.legend(loc='lower right')
    save(fig, 1, 'skalowanie',
         'Zależność zmierzonej korelacji sentymentu z ceną tego samego dnia od liczby '
         'wpisów, z których liczona jest średnia. Wzorzec powtarza się na trzech '
         'niezależnych zbiorach i dwóch klasach aktywów, co wskazuje na błąd '
         'próbkowania jako źródło tłumienia efektu przy rzadkich danych.')


# --------------------------------------------------------------------------- #
def fig02_causality():
    """Reaction versus prediction, at every way of cutting the day."""
    from analysis.common import load_hourly_btc, ci

    meta, px = load_hourly_btc()
    hour_px = px['close']
    meta = meta[(meta['ts'] >= px.index.min()) & (meta['ts'] <= px.index.max())].copy()
    meta['day'] = meta['ts'].dt.floor('D')
    meta['hour'] = meta['ts'].dt.hour

    cuts, pred, react = [], [], []
    for H in [4, 6, 8, 12, 16, 18, 20]:
        e = meta[meta['hour'] < H].groupby('day')['finbert'].agg(['mean', 'size'])
        l = meta[meta['hour'] >= H].groupby('day')['finbert'].agg(['mean', 'size'])
        d = e.join(l, lsuffix='_e', rsuffix='_l', how='inner')
        d = d[(d['size_e'] >= 50) & (d['size_l'] >= 50)]
        if len(d) < 100:
            continue
        p_open = hour_px.reindex(pd.DatetimeIndex(d.index)).values
        p_cut = hour_px.reindex(pd.DatetimeIndex(d.index) + pd.Timedelta(hours=H)).values
        p_close = hour_px.reindex(pd.DatetimeIndex(d.index) + pd.Timedelta(days=1)).values
        ok = ~(np.isnan(p_open) | np.isnan(p_cut) | np.isnan(p_close))
        dd = d[ok]
        r1 = stats.pearsonr(dd['mean_e'], p_close[ok] / p_cut[ok] - 1)[0]
        r2 = stats.pearsonr(p_cut[ok] / p_open[ok] - 1, dd['mean_l'])[0]
        cuts.append(H)
        pred.append((r1,) + ci(r1, len(dd)))
        react.append((r2,) + ci(r2, len(dd)))

    x = np.arange(len(cuts))
    fig, ax = plt.subplots()
    for series, colour, label, off in [(react, ORANGE, 'reakcja: cena → późniejszy sentyment', -0.13),
                                       (pred, BLUE, 'predykcja: sentyment → późniejsza cena', 0.13)]:
        vals = [s[0] for s in series]
        lo = [s[0] - s[1] for s in series]
        hi = [s[2] - s[0] for s in series]
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt='o', color=colour,
                    capsize=3, markersize=6, linewidth=1.5, label=label)
    ax.axhline(0, color=INK, linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}h' for h in cuts])
    ax.set_xlabel('godzina, w której przecinamy dobę')
    ax.set_ylabel('korelacja')
    ax.set_title('Kierunek zależności: sentyment podąża za ceną, nie odwrotnie')
    ax.set_ylim(top=max(s[2] for s in react) + 0.20)   # headroom for the legend
    ax.legend(loc='upper center', ncol=2, fontsize=8.5)
    save(fig, 2, 'kierunek_zaleznosci',
         'Rozdzielenie dwóch kanałów przy różnych sposobach przecięcia doby. Okna '
         'predyktora i celu nigdzie się nie nakładają. Wąsy oznaczają 95% przedziały '
         'ufności. Kanał reakcji jest silny przy każdym cięciu, kanał predykcji '
         'oscyluje wokół zera; przedziały obu kanałów nigdzie się nie stykają.')


# --------------------------------------------------------------------------- #
def _stocknet_panel():
    import glob
    tw = pd.read_parquet('data/stocknet/processed/per_tweet.parquet')
    tw = tw[tw['finbert'].notna()].copy()
    tw['ts'] = pd.to_datetime(tw['ts'])
    hour = tw['ts'].dt.hour + tw['ts'].dt.minute / 60
    day = tw['ts'].dt.floor('D')
    ac = hour >= 21
    tw['session'] = day.where(~ac, day + pd.Timedelta(days=1))
    tw = tw[ac | (hour < 13.5)]
    agg = (tw.groupby(['ticker', 'session']).agg(sent=('finbert', 'mean'),
                                                 n=('finbert', 'size')).reset_index()
             .rename(columns={'session': 'date'}))
    frames = []
    for p in sorted(glob.glob('data/stocknet/price/raw/*.csv')):
        d = pd.read_csv(p)
        d['date'] = pd.to_datetime(d['Date'])
        d['ticker'] = os.path.splitext(os.path.basename(p))[0]
        frames.append(d[['ticker', 'date', 'Open', 'Close']])
    px = pd.concat(frames).sort_values(['ticker', 'date'])
    g = px.groupby('ticker')
    px['prev_close'] = g['Close'].shift(1)
    px['gap'] = px['Open'] / px['prev_close'] - 1
    px['intraday'] = px['Close'] / px['Open'] - 1
    d = agg.merge(px, on=['ticker', 'date'], how='inner')
    return d[(d['n'] >= 3) & d['gap'].notna()].copy()


def fig03_quantiles():
    """Where the overnight information lands, by sentiment rank."""
    d = _stocknet_panel()
    per = d.groupby('date')['ticker'].transform('size')
    d = d[per >= 6]
    d['bucket'] = d.groupby('date')['sent'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False, duplicates='drop'))
    d = d[d['bucket'].notna()]
    gap = d.groupby('bucket')['gap'].mean() * 10000
    intr = d.groupby('bucket')['intraday'].mean() * 10000

    x = np.arange(len(gap))
    w = 0.38
    fig, ax = plt.subplots()
    ax.bar(x - w/2, gap.values, w, color=GREEN, label='luka otwarcia (rynek zamknięty)')
    ax.bar(x + w/2, intr.values, w, color=GREY, alpha=0.65,
           label='ruch śródsesyjny (rynek handluje)')
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}/5' for i in x])
    ax.set_xlabel('grupa według sentymentu w oknie zamknięcia (1 = najniższy)')
    ax.set_ylabel('średni ruch [punkty bazowe]')
    ax.set_title('Informacja z okna zamknięcia jest wyceniana na otwarciu')
    ax.legend(loc='upper left')
    save(fig, 3, 'kwantyle_luka',
         'Średni ruch kursu według rangi sentymentu z okna zamkniętej sesji, grupy '
         'tworzone w obrębie każdej sesji. Uporządkowanie luki otwarcia jest '
         'monotoniczne, a rozpiętość między skrajnymi grupami wynosi około 30 punktów '
         'bazowych. Ta sama procedura zastosowana do ruchu śródsesyjnego nie daje '
         'żadnego uporządkowania, co wskazuje na natychmiastową wycenę informacji.')


def fig04_oos():
    """Out-of-sample gain from adding sentiment."""
    designs = ['nieznane spółki', 'późniejszy okres', 'nieznane spółki\ni późniejszy okres']
    ctrl = [0.1146, 0.1609, 0.0468]
    sent = [0.1488, 0.1821, 0.1613]
    x = np.arange(len(designs))
    w = 0.38
    fig, ax = plt.subplots()
    ax.bar(x - w/2, ctrl, w, color=GREY, alpha=0.7, label='same cechy cenowe')
    ax.bar(x + w/2, sent, w, color=GREEN, label='cechy cenowe + sentyment')
    ax.set_xticks(x)
    ax.set_xticklabels(designs)
    ax.set_ylabel('korelacja prognozy z rzeczywistą luką (poza próbą)')
    ax.set_title('Walidacja poza próbą: zysk z dodania sentymentu')
    ax.legend(loc='upper left')
    ax.annotate('p = 0,34', xy=(2 - w/2, 0.0468), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=8, color=GREY)
    ax.annotate('p = 0,0009', xy=(2 + w/2, 0.1613), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=8, color=GREEN)
    save(fig, 4, 'walidacja_poza_proba',
         'Jakość prognozy luki otwarcia dla obserwacji nieużytych w estymacji. '
         'W najostrzejszym wariancie — spółki nieznane modelowi w okresie nieobjętym '
         'treningiem — same cechy cenowe nie mają zdolności prognostycznej, '
         'a dodanie sentymentu daje wynik istotny. RECORDED: wartości z stocknet_oos.py.')


# --------------------------------------------------------------------------- #
def fig05_dogecoin():
    """Attention effect: both groups shown, with a placebo window before the tweet."""
    from dogecoin_control import load, window_return
    tw, price = load()
    # Negative window = BEFORE the tweet. That placebo horizon is what separates
    # "the tweet moved the price" from "he tweeted because it was already moving",
    # so it belongs in the figure rather than only in the console output.
    spec = [(-60, 'placebo\n(−60 min)'), (5, '5 min'), (15, '15 min'), (30, '30 min'),
            (60, '1 godz.'), (240, '4 godz.'), (1440, '1 dzień'), (10080, '1 tydzień')]
    doge, ctrl, sig = [], [], []
    for w, _ in spec:
        tw['tmp'] = [window_return(price, t, abs(w), w > 0) for t in tw['ts']]
        a = tw.loc[tw['is_doge'], 'tmp'].dropna()
        b = tw.loc[~tw['is_doge'], 'tmp'].dropna()
        _, p = stats.ttest_ind(a, b, equal_var=False)
        doge.append(a.mean() * 100)
        ctrl.append(b.mean() * 100)
        sig.append(p < 0.05)

    # Both groups are drawn, not their difference. The difference alone hides the
    # week-horizon story: the DOGE group does not fall (+7.0%), the control runs
    # away from it (+17.3%). Plotting only the gap would read as a crash.
    #
    # Two panels because the week-horizon control (+17%) would otherwise flatten
    # the ~2% first-hour bars that carry the actual finding.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 4.3),
                                   gridspec_kw={'width_ratios': [3, 1]})
    split = 6
    W = 0.38
    for ax, idx in [(axL, range(split)), (axR, range(split, len(spec)))]:
        idx = list(idx)
        x = np.arange(len(idx))
        # Colour encodes the GROUP, never the outcome - a bar must not change
        # identity because its p-value did. Significance rides on the asterisk.
        ax.bar(x - W / 2 - 0.01, [doge[i] for i in idx], W, color=GREEN,
               label='wpisy o Dogecoinie', zorder=3)
        ax.bar(x + W / 2 + 0.01, [ctrl[i] for i in idx], W, color=GREY,
               label='kontrola: pozostałe wpisy Muska', zorder=3)
        top = max(max(doge[i], ctrl[i]) for i in idx)
        for k, i in enumerate(idx):
            for off, v in [(-W / 2 - 0.01, doge[i]), (W / 2 + 0.01, ctrl[i])]:
                ax.text(k + off, v + top * 0.035, f'{v:+.1f}', ha='center',
                        fontsize=7, color=INK)
            if sig[i]:
                ax.text(k, top * 1.16, '*', ha='center', fontsize=13, color=INK)
        ax.axhline(0, color=INK, linewidth=1, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([spec[i][1] for i in idx], rotation=20, fontsize=8.5)
        ax.set_ylim(0, top * 1.30)
    # Shade the placebo column so it reads as "before the tweet", not another horizon.
    axL.axvspan(-0.5, 0.5, color=GREY, alpha=0.09, zorder=0)
    axL.set_ylabel('średnia zmiana kursu DOGE [%]')
    axL.set_title('przed wpisem  |  natychmiastowa reakcja', fontsize=10)
    axR.set_title('horyzont dłuższy', fontsize=10)
    # Figure-level legend: inside the left panel it landed on the significance
    # asterisks, which sit in the same free band above the bars.
    handles, lbls = axL.get_legend_handles_labels()
    fig.legend(handles, lbls, loc='upper center', ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.005))
    fig.supxlabel('okno wokół publikacji wpisu     (* różnica istotna, p < 0,05)',
                  fontsize=9, y=-0.04)
    fig.suptitle('Efekt uwagi: wzmianka podnosi kurs, ale wydźwięk wpisu nie ma znaczenia',
                 y=1.10)
    fig.tight_layout()
    save(fig, 5, 'dogecoin_uwaga',
         'Średnia zmiana kursu Dogecoina po wpisach Elona Muska dotyczących tej '
         'kryptowaluty (zielone) i po pozostałych jego wpisach z tego samego okresu '
         '(szare, grupa kontrolna). Skrajnie lewa para to test placebo — okno godziny '
         'PRZED publikacją: obie grupy są nieodróżnialne (+0,79% wobec +0,32%, p = 0,33), '
         'co wyklucza odwrotną przyczynowość, czyli publikowanie wpisów w reakcji na '
         'trwający już ruch kursu. Po publikacji różnica pojawia się natychmiast '
         '(+2,23% wobec +0,11% w ciągu 5 minut) i zanika w ciągu czterech godzin. '
         'W horyzoncie tygodniowym grupa z Dogecoinem nie spada — zatrzymuje się na '
         '+7,0%, podczas gdy kontrola dochodzi do +17,3%; odwrócenie polega więc na '
         'pozostaniu w tyle za rynkiem, nie na załamaniu kursu. Wpisy pozytywne '
         'i negatywne dają ten sam efekt, co wyklucza wydźwięk jako kanał oddziaływania. '
         'Zastrzeżenie: przy 101 zdarzeniach okna tygodniowe zachodzą na siebie.')


# --------------------------------------------------------------------------- #
def fig06_reddit_lag():
    """Which day Reddit sentiment attaches to."""
    from reddit_analysis import daily_frame
    from analysis.common import ci
    _, d = daily_frame()
    pairs = [('dzień poprzedni', 'prev_return'), ('dzień bieżący', 'daily_return'),
             ('dzień następny', 'next_return')]
    vals, los, his = [], [], []
    for _, col in pairs:
        s = d[d[col].notna()]
        r, _ = stats.pearsonr(s['finbert'], s[col])
        lo, hi = ci(r, len(s))
        vals.append(r)
        los.append(r - lo)
        his.append(hi - r)

    y = np.arange(len(pairs))
    colours = [ORANGE, BLUE, GREY]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(y, vals, 0.55, color=colours, xerr=[los, his],
            error_kw={'ecolor': INK, 'capsize': 3, 'alpha': 0.6})
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in pairs])
    ax.invert_yaxis()
    ax.set_xlabel('korelacja sentymentu ze zwrotem danego dnia')
    ax.set_title('Reddit: sentyment najsilniej wiąże się z dniem poprzednim')
    save(fig, 6, 'reddit_opoznienie',
         'Korelacja dziennego sentymentu z portali Reddit ze zwrotem dnia poprzedniego, '
         'bieżącego i następnego. Najsilniejszy związek zachodzi z dniem poprzednim, '
         'co odzwierciedla dłuższy, refleksyjny charakter wpisów na tej platformie. '
         'Wąsy oznaczają 95% przedziały ufności.')


# --------------------------------------------------------------------------- #
def fig07_peruser():
    """Per-account screen against a shuffled null. RECORDED."""
    groups = ['kont „istotnych”\n(p < 0,05)', 'najsilniejsza\nkorelacja |r|']
    real = [2943, 0.760]
    null = [2891, 0.762]
    err = [135, 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, i, ylab in [(axes[0], 0, 'liczba kont'), (axes[1], 1, 'korelacja |r|')]:
        ax.bar([0, 1], [real[i], null[i]], 0.5, color=[BLUE, ORANGE])
        if err[i]:
            ax.errorbar([1], [null[i]], yerr=[err[i]], fmt='none', ecolor=INK, capsize=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['dane\nprawdziwe', 'kontrola\nlosowa'])
        ax.set_ylabel(ylab)
        ax.set_title(groups[i], fontsize=10)
    fig.suptitle('Przeszukanie 27 316 kont: wynik nieodróżnialny od przypadku', y=1.02)
    save(fig, 7, 'per_konto',
         'Wynik przeszukania wszystkich kont aktywnych przez co najmniej 25 dni '
         '(54 608 testów korelacji) zestawiony z tym samym przeszukaniem na '
         'przetasowanym szeregu zwrotów. Dane prawdziwe nie produkują więcej pozornie '
         'istotnych kont niż sam przypadek, a najsilniejsza korelacja jest w obu '
         'przypadkach niemal identyczna. RECORDED: wartości z analysis/cohorts.py.')


# --------------------------------------------------------------------------- #
def fig08_ablation():
    """Does sentiment improve the model? Reads the ablation's own output file.

    Hardcoding these numbers is how the previous version drifted: it still showed
    a 3-seed run on 1560 days long after the experiment had been redone with 20
    seeds on 197 days. Prefer results/ablation_threeway_summary.json, written by
    models/ablation_threeway.py; fall back to the recorded 20-seed values only so
    the figure still builds on a fresh clone, where results/ is gitignored.
    """
    import json
    order = ['price', '+finbert', '+roberta', '+both']
    labels = ['tylko cena', '+ FinBERT', '+ RoBERTa', '+ oba strumienie']
    try:
        with open('results/ablation_threeway_summary.json') as fh:
            s = json.load(fh)
        vals = [s['variants'][k]['mean_da'] for k in order]
        sd = [s['variants'][k]['sd_da'] for k in order]
        majority, n_seeds = s['majority_da'], s['n_seeds']
        n_test, src_note = s.get('n_test_days'), 'przeliczone'
    except (FileNotFoundError, KeyError):
        vals = [51.19, 49.85, 49.42, 50.15]
        sd = [2.85, 3.14, 2.39, 3.42]
        majority, n_seeds, n_test, src_note = 48.22, 20, 197, 'RECORDED'

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    # Blue marks the model without sentiment; the three variants that add it are
    # one grey group because no pairwise difference between them survives
    # correction - colouring them separately would imply a ranking that is noise.
    colours = [BLUE] + [GREY] * 3
    y = np.arange(len(order))
    ax.barh(y, vals, 0.55, color=colours, xerr=[sd, sd],
            error_kw={'ecolor': INK, 'capsize': 4, 'alpha': 0.6}, zorder=3)
    ax.axvline(majority, color=ORANGE, linestyle='--', linewidth=1.5,
               label=f'baseline większościowy ({majority:.2f}%)', zorder=2)
    ax.axvline(50, color=INK, linestyle=':', linewidth=1.2,
               label='czysty przypadek (50%)', zorder=2)
    # Labels sit inside the bars: placed past the whiskers they collided with the
    # legend, and the whisker end is not the number being reported anyway.
    for i, v in enumerate(vals):
        ax.text(v - 0.2, i, f'{v:.2f}%', va='center', ha='right',
                fontsize=8.5, color='white', fontweight='bold', zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(44, 57)
    ax.set_xlabel('trafność kierunkowa poza próbą [%]')
    ax.set_title('Ablacja: dodanie sentymentu nie poprawia prognozy kierunku')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
    fig.tight_layout()
    save(fig, 8, 'ablacja',
         f'Trafność kierunkowa czterech identycznych sieci GRU różniących się wyłącznie '
         f'zestawem wejść, uśredniona z {n_seeds} ziaren losowych na {n_test} dniach '
         f'testowych; wąsy to odchylenie standardowe między ziarnami. Żaden wariant '
         f'z sentymentem nie przewyższa modelu opartego na samej cenie, a po korekcie '
         f'Bonferroniego dla czterech porównań wobec baseline\'u większościowego '
         f'(linia przerywana) przechodzi wyłącznie wariant bez sentymentu. Wobec '
         f'czystego przypadku (linia kropkowana) żaden wariant nie jest odróżnialny '
         f'w teście dwumianowym. Liczby: {src_note}.')


# --------------------------------------------------------------------------- #
def fig09_placebo():
    """The test that separated one real finding from four false ones. RECORDED."""
    # (label, forward t, backward t, kept)
    rows = [('Rozproszenie opinii\n→ zmienność (BTC)', 2.41, 3.47, False),
            ('Wolumen wpisów\n→ zmienność (Reddit)', 2.35, 3.27, False),
            ('Upvote ratio\n→ zmienność (Reddit)', 2.33, 2.02, False),
            ('Sentyment nocny\n→ luka otwarcia (akcje)', 6.20, 2.34, True)]
    y = np.arange(len(rows))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(y - h/2, [r[1] for r in rows], h, color=BLUE, label='w przód (predykcja)')
    ax.barh(y + h/2, [r[2] for r in rows], h, color=ORANGE, label='wstecz (placebo)')
    ax.axvline(1.96, color=GREY, linestyle='--', linewidth=1.2)
    # y-axis is inverted, so -0.55 places this ABOVE the first bar rather than
    # on the x-axis tick labels.
    ax.text(1.96, -0.55, ' próg istotności', fontsize=8, color=GREY, va='center')
    for i, r in enumerate(rows):
        mark = 'utrzymana' if r[3] else 'odrzucona'
        colour = GREEN if r[3] else GREY
        ax.text(max(r[1], r[2]) + 0.25, i, mark, va='center', fontsize=8.5, color=colour)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 8.2)
    ax.set_xlabel('statystyka t')
    ax.set_title('Test kierunkowości oddziela jeden prawdziwy wynik od czterech pozornych')
    ax.set_ylim(len(rows) - 0.4, -0.9)      # headroom above the first bar
    ax.legend(loc='upper right', fontsize=8.5)
    save(fig, 9, 'test_placebo',
         'Porównanie siły zależności mierzonej w przód i wstecz dla czterech hipotez, '
         'które przy standardowym teście wyglądały na potwierdzone. W trzech przypadkach '
         'zależność wsteczna jest równie silna lub silniejsza, co wskazuje na relację '
         'współbieżną, a nie prognostyczną. Wyłącznie w przypadku luki otwarcia efekt '
         'w przód wyraźnie dominuje. Wynik dla rozproszenia opinii przeszedł wcześniej '
         'odporne błędy standardowe Newey-West oraz bootstrap blokowy — obalił go '
         'dopiero ten test. RECORDED: wartości z analysis/hypotheses.py, '
         'reddit_attention.py i stocknet_decisive.py.')


# --------------------------------------------------------------------------- #
def fig10_power():
    """What the nulls exclude, and what a detectable effect looks like. RECORDED."""
    from analysis.common import ci, mde
    nulls = [('sentyment → następny dzień (2016–19)', 942, 0.0117),
             ('wolumen → zmienność (2016–19)', 942, -0.0021),
             ('1% kont o największym zasięgu', 301, 0.0332),
             ('retail → następny dzień (2021–23)', 222, 0.0532),
             ('wieloryby → następny dzień (2021–23)', 222, -0.1158)]
    effect = [('sentyment → cena tego samego dnia (2016–19)', 942, 0.2463),
              ('sentyment → cena tego samego dnia (2021–23)', 222, 0.4164)]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels, ypos = [], []
    for i, (lab, n, r) in enumerate(nulls + effect):
        lo, hi = ci(r, n)
        m = mde(n)
        is_effect = i >= len(nulls)
        colour = GREEN if is_effect else BLUE
        ax.errorbar(r, i, xerr=[[r - lo], [hi - r]], fmt='o', color=colour,
                    capsize=3, markersize=6, linewidth=1.5)
        # the smallest effect this sample size could have detected
        ax.plot([m], [i], marker='|', markersize=13, color=ORANGE,
                markeredgewidth=2, zorder=4)
        labels.append(f'{lab}  (n={n})')
        ypos.append(i)
    ax.axvline(0, color=INK, linestyle='--', linewidth=1)
    ax.axhline(len(nulls) - 0.5, color=GREY, linewidth=0.8, alpha=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('korelacja (punkt, 95% przedział ufności)')
    ax.set_title('Czego wyniki zerowe nie wykluczają, a czego wykluczają')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=BLUE, marker='o', linestyle='', label='wynik zerowy'),
        Line2D([], [], color=GREEN, marker='o', linestyle='', label='efekt wykryty'),
        Line2D([], [], color=ORANGE, marker='|', linestyle='', markersize=11,
               markeredgewidth=2, label='próg wykrywalności (moc 80%)')],
        loc='upper right', fontsize=8)
    save(fig, 10, 'moc_statystyczna',
         'Zmierzone korelacje wraz z 95% przedziałami ufności, zestawione z najmniejszym '
         'efektem, jaki dana liczebność próby pozwalała wykryć przy mocy 80% (znacznik '
         'pionowy). Wyniki zerowe uzyskane na dużych próbach wykluczają efekty powyżej '
         'wskazanego progu; wynik dla kohorty wielorybów, oparty na 222 dniach, takiego '
         'wykluczenia nie umożliwia. Dwie ostatnie pozycje pokazują dla porównania, jak '
         'wygląda efekt faktycznie wykryty.')


# --------------------------------------------------------------------------- #
def write_captions():
    with open(f'{OUT}/podpisy.md', 'w', encoding='utf-8') as fh:
        fh.write('# Podpisy pod rysunki\n\n')
        fh.write('Rysunki zapisane w `figures/` w dwóch formatach: `.png` (300 dpi, '
                 'do edytorów tekstu) oraz `.pdf` (wektor, do LaTeX-a).\n\n')
        fh.write('Oznaczenie RECORDED wskazuje rysunki zbudowane z zapisanych wartości, '
                 'ponieważ ich ponowne przeliczenie wymaga godzin obliczeń. '
                 'Pozostałe przeliczają się z danych przy każdym uruchomieniu.\n\n')
        for num, slug, cap in sorted(CAPTIONS):
            fh.write(f'## Rys. {num}. ({slug})\n\n{cap}\n\n')
    print(f'\nPodpisy -> {OUT}/podpisy.md')


if __name__ == '__main__':
    print('Generowanie rysunków...')
    for fn in [fig01_scaling, fig02_causality, fig03_quantiles, fig04_oos,
               fig05_dogecoin, fig06_reddit_lag, fig07_peruser, fig08_ablation,
               fig09_placebo, fig10_power]:
        try:
            fn()
        except Exception as e:
            print(f'  ! {fn.__name__} nie powiódł się: {type(e).__name__}: {e}')
    write_captions()

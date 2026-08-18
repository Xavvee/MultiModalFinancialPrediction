# Odtwarzanie wyników

Każda liczba w dzienniku (`docs/dziennik.html`) i w pracy pochodzi z jednej
z poniższych komend. Zbiory danych są w `.gitignore` — sekcja „Dane" opisuje,
skąd je wziąć.

## Kolejność

Potoki przygotowania danych trzeba uruchomić raz; analizy czytają już gotowe
pliki `.parquet` i są szybkie.

```
# 1. potok danych (raz na zbiór)
python data_loader.py           # surowe CSV -> clean_tweets
python feature_engineer.py      # wagi zaangażowania / autorytetu
python market_processor.py      # cechy rynkowe z yfinance
python nlp_processor.py         # sentyment (korzysta z cache, patrz niżej)
python tools/build_parquet.py   # kompaktowy per_tweet.parquet
python tools/etl_intraday.py    # ETL zachowujący godziny (do testu kierunkowości)
python tools/fetch_hourly.py    # ceny godzinowe BTC z Binance

# 2. analizy
python -m analysis.verification # bramki 1-3: czy zbiorowi można ufać
python -m analysis.forensics    # wykrywanie sfabrykowanych dat
python -m analysis.causality    # KLUCZOWE: kierunek zależności (BTC)
python -m analysis.power        # moc statystyczna i przedziały ufności
python -m analysis.cohorts      # kohorty zasięgu + przegląd per konto
python -m analysis.hypotheses   # hipotezy alternatywne, wszystkie odrzucone

# 4. sieć GRU (gałąź old_dataset albo main - wariant regresyjny)
python -m models.ablation_threeway 20   # 4 warianty wejść x 20 ziaren + testy istotności
python -m models.lazy_predictor_test    # bezposredni test zapadniecia sie prognoz
python main.py                          # pelny benchmark + tabela wspolnego okna

# 5. stabilnosc walidacji kroczacej (WYLACZNIE galaz new_dataset)
python -m models.gru_stability 10       # 5 okien x 10 powtorzen + konfundy

# 3. akcje (stocknet)
python stocknet_loader.py       # 29 tys. plików JSON -> parquet
python stocknet_nlp.py          # sentyment
python stocknet_analysis.py     # bramki + główny test luki otwarcia
python stocknet_decisive.py     # czy wynik przeżywa kontrolę przeszłości
python stocknet_robustness.py   # 6 rodzin kontroli odporności
python stocknet_permutation.py  # test permutacyjny (500 losowań)

# 4. kontrola pozytywna (dogecoin)
python dogecoin_sentiment.py    # sentyment wpisów Muska
python dogecoin_control.py      # badanie zdarzeniowe + placebo

# 5. benchmark modeli cenowych
python main.py                  # wszystkie modele + baseline większościowy
```

## Gdzie znaleźć konkretny wynik

| Wynik w pracy | Komenda |
|---|---|
| Wykrycie sfabrykowanego zbioru | `analysis.forensics` |
| Prawo skalowania (korelacja rośnie z gęstością) | `analysis.verification` |
| Kierunek zależności na BTC (reakcja 0,41 / predykcja 0,02) | `analysis.causality` |
| Moc: wykluczamy efekt powyżej 0,076 | `analysis.power` |
| Brak wpływowych kont (test permutacyjny) | `analysis.cohorts` |
| Rozproszenie opinii — obalone przez placebo | `analysis.hypotheses` |
| Boty, horyzonty, reszta, sprzężenie zwrotne | `analysis.hypotheses` |
| Porównanie FinBERT vs RoBERTa | `analysis.hypotheses` |
| Sentyment nocny → luka otwarcia (t=+6,2) | `stocknet_decisive.py` |
| 7,7 odchylenia od rozkładu zerowego | `stocknet_permutation.py` |
| Efekt Muska +2,4 p.p. w 15 minut | `dogecoin_control.py` |
| Baseline większościowy | `models/majority_baseline.py` przez `main.py` |
| Ablacja: 4 warianty wejść, 20 ziaren | `models.ablation_threeway` (gałąź `main`/`old_dataset`) |
| Korekta Bonferroniego + test dwumianowy | `models.ablation_threeway` |
| Lazy Predictor odrzucony (sd 0,40–1,33) | `models.lazy_predictor_test` |
| Stabilność walidacji kroczącej (49,4% ± 1,9) | `models.gru_stability` (gałąź `new_dataset`) |
| Konfund: rozmiar zbioru treningowego r=+0,918 | `models.gru_stability` |
| Tabela wspólnego okna + pokrycie | `models/benchmark_table.py` przez `main.py` |
| Baseline persystencji (mierzalny odpowiednik RW) | `models/persistence_baseline.py` przez `main.py` |

## Cache sentymentu

`data/sentiment_cache.parquet` mapuje skrót tekstu na wyniki obu modeli.
Sentyment zależy wyłącznie od treści, więc identyczny tekst nigdy nie jest
liczony dwa razy — także między zbiorami. Przy przetwarzaniu poprawionego
zbioru 2021–23 dało to 96% trafień i oszczędziło praktycznie cały czas GPU.
Usunięcie pliku wymusza przeliczenie od zera (godziny na GPU).

## Układ gałęzi

Repozytorium ma pięć gałęzi, bo każda odpowiada innemu korpusowi. Kod analityczny
jest wspólny, różnią się dane i wariant sieci.

| Gałąź | Korpus | Wariant sieci | Co produkuje |
|---|---|---|---|
| `main` | Bitcoin 2016–2019 | GRU regresyjny (MSE) | benchmark, ablacja, testy placebo, pełna analiza |
| `old_dataset` | Bitcoin 2016–2019 | GRU regresyjny (MSE) | to samo co `main`; gałąź źródłowa tych skryptów |
| `new_dataset` | Bitcoin 2021–2023 | GRU klasyfikacyjny (sigmoid + BCE) | walidacja krocząca, stabilność między ziarnami |
| `stocknet` | 88 spółek 2014–2016 | — | jedyny wynik pozytywny: luka otwarcia |
| `reddit` | Reddit 2022 | — | porównanie platform |

**Główny wynik pracy** (ablacja na 20 ziarnach) odtwarza się na `main` albo
`old_dataset`. **Walidacja krocząca** wymaga `new_dataset`, bo tamtejszy wariant
sieci jest klasyfikatorem z osobnymi strumieniami wieloryby/ulica.

## Znane ograniczenia kodu

- Modele nie są seedowane w `main.py`, więc pojedynczy przebieg waha się o kilka
  punktów procentowych (LSTM: odchylenie 4,07 p.p. między ziarnami; GRU: 3,42).
  Do pracy należy cytować rozkłady z `ablation_threeway.py`, nie pojedynczy
  wiersz z tabeli benchmarku.
- Tabela wspólnego okna ma małe `n` (145 dni dla BTC 2016–19), bo przecięcie okien
  testowych ogranicza najdłuższe okno wsteczne (`lstm_stationary`, LOOK_BACK=60).
  Skrypt drukuje pod tabelą margines błędu przy tym `n`.
- Ważenie zaangażowaniem (`feature_engineer.py`) trafia wyłącznie do sieci GRU.
  Cała analiza w `analysis/` używa zwykłej średniej z surowych ocen per wpis.


## Dane

Zbiory nie są w repozytorium. Do pobrania:

- **BTC 2016–2019** — Kaggle `alaix14/bitcoin-tweets-20160101-to-20190329`
- **BTC 2021–2023** — Kaggle `kaushiksuresh147/bitcoin-tweets`
  (**nie** `pokeash/bitcoin-tweets-dataset-20252026` — ma sfabrykowane daty,
  patrz `analysis/forensics.py`)
- **Akcje** — `github.com/yumoxu/stocknet-dataset` (Xu & Cohen, ACL 2018)
- **Dogecoin** — Kaggle `johnsmith44/dogecoin-price-data-elon-musks-tweets-2021`
- **Ceny godzinowe BTC** — pobierane skryptem z publicznego API Binance

`data/evidence/` zawiera po 500 wierszy ze zbioru sfabrykowanego i oryginalnego,
żeby porównanie dało się pokazać bez trzymania 2 GB.

## Uwaga o wielokrotnym testowaniu

W całym badaniu przeprowadzono ponad 80 testów na powiązanych danych. Liczba jest
raportowana jawnie w dzienniku. Wynik dla akcji był hipotezą postawioną z góry,
a jego wzorzec (efekt w luce otwarcia, brak śródsesyjnego) przewidziano przed
pomiarem. Dwa wyniki, które wyglądały pozytywnie, zostały odrzucone testem
placebo — oba opisane w `analysis/hypotheses.py`.

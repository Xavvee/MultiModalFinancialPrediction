# MultiModalFinancialPrediction — kontekst projektu (praca magisterska)

Repo: https://github.com/Xavvee/MultiModalFinancialPrediction
Autor: Hubert. Ten plik ma dawać każdej sesji Claude (Code/Cowork) pełny kontekst bez ponownego tłumaczenia od zera.

## Cel pracy

Analiza wpływu wpisów w mediach społecznościowych na wartość aktywów finansowych (BTC-USD, docelowo też akcje/indeksy). Hipoteza badawcza: dodanie sygnału sentymentu poprawia Directional Accuracy (DA) prognoz względem modeli opartych wyłącznie na cenie. Drugi cel: wskazanie **konkretnych użytkowników** (per-user, nie tylko per-tweet) o największym wpływie na kierunek rynku.

## WAŻNE: dwa datasety, dwa branche

Projekt operuje na dwóch różnych zbiorach tweetów o Bitcoinie, stąd rozjazdy między branchami:

1. **`old_dataset` branch** — dataset Kaggle [alaix14/bitcoin-tweets-20160101-to-20190329](https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329) (~16M tweetów, 2016–2019). **Ten branch ma działający `models/gru_multimodal.py`** (zweryfikowane) — GRU 64→32, BatchNorm, Dropout 0.3, look-back 14, regresja na `daily_return`, wejście: `daily_return, momentum_3d, volatility_7d, finbert_sentiment, roberta_sentiment`. `utils.get_data` ma spójny zakres dat (2017-01-01–2019-12-01) dopasowany do datasetu. **To jest branch, który Hubert chce doprowadzić do stanu końcowego jako pierwszy priorytet.**
2. **`main` branch** — nowy dataset Kaggle [pokeash/bitcoin-tweets-dataset-20252026](https://www.kaggle.com/datasets/pokeash/bitcoin-tweets-dataset-20252026) (2025–2026). Praca w toku, `models/gru_multimodal.py` jeszcze nie istnieje na tym branchu (stąd błąd importu opisany niżej dla main — to nie błąd, tylko stan "w budowie").

Gdy poniżej mowa o "Torze A/B", dotyczy to obu branchy (struktura identyczna), różni je tylko dataset tweetów i zakres dat w `utils.py`.

## Stan repo — DWA NIEPOŁĄCZONE TORY

### Tor A: benchmark cenowy (main.py + models/)
`main.py` uruchamia dla ASSETS = ["BTC-USD", "ETH-USD", "^GSPC"]:
1. `random_walk.py` — baseline naiwny
2. `arima.py` — ARIMA na cenach
3. `arima_stationary.py` — ARIMA na zwrotach %
4. `lstm.py` — LSTM na cenach
5. `lstm_stationary.py` — LSTM na zwrotach %
6. `gru_multimodal.py` — **NIE ISTNIEJE W REPO** (patrz "Znane problemy")
7. `dashboard.py` — raport porównawczy

Dobre decyzje metodologiczne już tu są: baseline naiwny, porównanie modelowania cen vs zwrotów (stacjonarność), metryka DA obok RMSE. `utils.get_data()` i modele są ticker-agnostyczne — działają dla dowolnego tickera yfinance, więc rozszerzenie o kolejne aktywa jest tanie.

### Tor B: pipeline sentymentu (root: data_loader.py → feature_engineer.py → market_processor.py → nlp_processor.py)
1. `data_loader.py` — ETL surowego CSV tweetów (chunki, filtrowanie języka FastText, czyszczenie tekstu)
2. `feature_engineer.py` — waga zaangażowania: `log1p(likes + 20*retweets + 15*replies)`
3. `market_processor.py` — pobiera OHLCV z yfinance, liczy daily_return, momentum_3d/7d, volatility_7d, volume_change
4. `nlp_processor.py` — dual-stream sentyment: FinBERT (ProsusAI/finbert) + Twitter-RoBERTa (cardiffnlp), batch inference na GPU (torch_directml/AMD), checkpointing co 100k tweetów, merge z danymi rynkowymi → `full_dataset_weighted.pkl`
5. `sentiment_ab_test.py` — porównanie sentymentu ważonego vs nieważonego (A/B)

## Znane problemy (do naprawienia w pierwszej kolejności)

1. **Brakujący model na `main`**: `main.py` importuje `models.gru_multimodal`, którego nie ma na branchu `main` (zweryfikowane — 404). To nie jest bug do "naprawy" w sensie regresji — to po prostu niedokończona migracja na nowy dataset. Najprostsza droga: skopiować działającą wersję z `old_dataset` i dostosować do nowego zakresu dat/datasetu.
2. **Kohorty Wieloryby/Ulica vs cel "per-user".** `magisterka.txt` opisuje segmentację po `user_verified` + `user_followers > 100000` (waga 5.0 vs 1.0), ale **`data_loader.py` na obu branchach wczytuje z CSV tylko kolumny `user, timestamp, replies, likes, retweets, text`** — nie ma tam `user_followers` ani `user_verified`. Do potwierdzenia na desktopie: czy surowe pliki Kaggle w ogóle mają takie kolumny w schemacie, czy trzeba je dociągnąć/oszacować inaczej. Hubert doprecyzował, że cel to identyfikacja **konkretnych wpływowych użytkowników (per-user)**, nie tylko sztywny podział whale/retail — czyli docelowo potrzebna jest agregacja cech per `user` (częstotliwość, zaangażowanie, sentyment, korelacja z ruchem ceny) i ranking/scoring wpływu, a nie tylko binarna kohorta. To wymaga osobnego projektu feature engineeringu, nie tylko dopisania kolumny.
3. **Możliwy data leakage**: w `lstm_stationary.py`, `dashboard.py` i `gru_multimodal.py` (potwierdzone też na `old_dataset`) `MinMaxScaler.fit_transform()` jest wywoływany na całym szeregu PRZED podziałem train/test — scaler "widzi" statystyki zbioru testowego. Powinno być `fit` tylko na train, `transform` osobno na test. Recenzenci praktycznie zawsze to wyłapują — warto naprawić przy okazji domykania `old_dataset`.
4. `requirements.txt` nie zawiera `torch`, `transformers`, `torch-directml`, `fasttext`, `huggingface_hub` — zależności Toru B nie są udokumentowane/zainstalowane razem z Torem A.
5. `gru_multimodal.py` (oba branche) działa tylko dla BTC-USD — dla ETH-USD/^GSPC jest pomijany (brak datasetu NLP dla tych aktywów). To świadome ograniczenie, nie bug.

## Środowisko

Trening lokalny na stacji z GPU AMD Radeon, akceleracja przez `torch_directml` (nie CUDA). Dataset tweetów: ~3.5 mln, lata 2025–2026.

## Roadmap (zaktualizowana wg decyzji Huberta 2026-07-25)

**Faza 0 — domknąć `old_dataset` do stanu końcowego (BIEŻĄCY PRIORYTET):**
- Sprawdzić na desktopie realny schemat surowego CSV (alaix14) — czy są jakiekolwiek pola przydatne do identyfikacji użytkownika poza `user` (np. followers/verified w innej kolumnie niż zakładano).
- Naprawić scaler leakage w `lstm_stationary.py`, `dashboard.py`, `gru_multimodal.py`.
- Zaprojektować i dodać per-user feature engineering (patrz niżej) zamiast sztywnej kohorty whale/retail.
- Domknąć end-to-end run `main.py` na branchu `old_dataset` z pełnym raportem/dashboardem.

**Faza 0.5 — per-user influence scoring (nowy kierunek zamiast prostego whale/retail):**
Cel: wskazać konkretnych użytkowników o największym wpływie, nie tylko kategorię. Do zaprojektowania:
- Zestaw cech per `user`: liczba tweetów, średnie zaangażowanie (`engagement_weight`), średni sentyment, ew. lead-lag korelacja sentymentu danego usera z późniejszym ruchem ceny.
- Metoda scoringu/rankingu (np. ablacja — usuwanie top-N userów i sprawdzanie spadku DA modelu; albo korelacja/Granger causality per user; liczba obserwacji per user w tym datasecie może być zbyt mała dla części metod — do zweryfikowania na realnych danych).
- To jest badawczo najciekawszy, ale i najbardziej niepewny element pracy — warto zacząć od prostego rankingu po zaangażowaniu, potem iterować.

**Faza 1 — spójny benchmark (po domknięciu Fazy 0):**
Jeden przebieg `main.py`, wszystkie modele na tym samym oknie czasowym, jeden dashboard DA/RMSE. Dodać test istotności różnic (np. Diebold-Mariano dla DA) — samo porównanie wykresów nie obroni się na recenzji.

**Faza 2 — powtórzyć Fazę 0–1 na `main` (nowy dataset 2025–2026):**
Dopiero po tym jak `old_dataset` jest w pełni domknięty i sprawdzony — przenieść te same poprawki/metodykę na nowy dataset.

**Faza 3 — rozszerzenie o S&P 500 (blokowane wyborem datasetu tweetów):**
Otwarte: indeks (^GSPC — już jest w Torze A) czy pojedyncze spółki. Tor A (cenowy) już działa dla dowolnego tickera bez zmian — problemem jest tylko dataset tekstowy.

Research (2026-07-25) — kandydaci na dataset tweetów o spółkach:

1. **stocknet-dataset (Xu & Cohen, ACL 2018)** — https://github.com/yumoxu/stocknet-dataset — **REKOMENDOWANY**. 88 spółek z 9 sektorów (top-cap z każdego sektora + wszystkie Conglomerates) — de facto reprezentatywna próbka S&P, gotowa i już przemyślana przez badaczy, więc nie trzeba samemu dobierać spółek. Tweety 2014-01-01–2016-01-01, **raw JSON = pełny obiekt Twitter API — zweryfikowałem, że zawiera `followers_count` i `verified`** (dokładnie te pola, których opis w magisterka.txt wymaga i których brakuje w obecnym datasecie BTC). Preprocessed wersja ma `user_id_str` — gotowe do agregacji per-user. Zawiera też dane cenowe (Yahoo Finance) już dopasowane. Publikacja podaje baseline'y (RAND, ARIMA, RandomForest, TSLDA, HAN — accuracy 50.9–57.6%, Tabela 1 w papierze) — można się do nich odnieść w pracy jako punkt odniesienia. **Minus: umiarkowana skala jak na "duży" dataset — 26,614 próbek dzień-akcja w całym zbiorze (nie miliony tweetów jak obecny BTC dataset)** i dane są z 2014-2016, nie najświeższe.
2. **StephanAkkerman/stock-market-tweets-data** (HuggingFace, ~924k tweetów, 2020, tagi S&P500 + top 25 spółek) — dużo większa skala, ale **schemat to tylko `id, created_at, text`** — brak jakichkolwiek danych o użytkowniku (nawet likes/retweets). Bezużyteczny do per-user, można by co najwyżej zrobić czysty sentyment zagregowany po całym rynku.
3. **StephanAkkerman/crypto-stock-tweets** (HuggingFace, 8.02M wierszy, crypto+stock połączone) — jeszcze większa skala, ale **schemat to tylko `text, url`** (url = źródłowy dataset) — zero metadanych użytkownika. Ten sam problem co wyżej.
4. FNSPID (github.com/Zdong104/FNSPID_Financial_News_Dataset) — 15.7M artykułów news + 29.7M cen dla 4775 spółek S&P500, 1999-2023 — ogromna skala, ale to **wiadomości finansowe, nie media społecznościowe** (Bloomberg/Reuters/Benzinga), więc nie pasuje do tezy pracy ("wpływ mediów społecznościowych") i nie ma pojęcia "użytkownika" do per-user influence.

**Wniosek/napięcie do rozstrzygnięcia z Hubertem**: dostępne publicznie duże datasety tweetów o spółkach (setki tysięcy–miliony wierszy) nie mają metadanych użytkownika, a te z metadanymi użytkownika (stocknet) są średniej wielkości, nie "duże". Te dwa kryteria (wolumen vs. dane per-user) się wykluczają w publicznie dostępnych zbiorach — trzeba wybrać kompromis albo poszukać jeszcze głębiej (np. własny scraping przez Twitter/X API, co ma teraz mocno ograniczony darmowy dostęp).

**Faza 4 — walidacja wyników:**
Walk-forward / rolling window zamiast pojedynczego podziału 80/20, ablacje (model bez sentymentu vs z sentymentem vs z per-user scoring), testy istotności statystycznej.

## Decyzje podjęte (2026-07-25)

- Kohorty mają być **per-user**, nie per-tweet — cel to wskazanie konkretnych wpływowych kont, nie tylko kategorii.
- S&P 500 (indeks vs spółki) — decyzja odłożona do momentu znalezienia odpowiedniego datasetu tekstowego.
- Priorytet: dokończyć branch `old_dataset` do stanu końcowego, zanim ruszy się dalej (S&P 500 / nowy dataset na `main`).
- Hubert doda pełny dostęp do plików (repo + surowe dane) po przejściu na komputer stacjonarny — wtedy można zweryfikować realny schemat danych zamiast zgadywać z kodu na GitHubie.
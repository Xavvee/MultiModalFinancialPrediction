# Analiza wpływu wpisów w mediach społecznościowych na zmiany wartości aktywów finansowych

## 📌 Cel Projektu
Celem pracy jest opracowanie hybrydowego systemu prognozowania cen aktywów (Kryptowaluty, Indeksy Giełdowe), który łączy:
1.  **Analizę Techniczną:** Historyczne dane cenowe (OHLCV).
2.  **Analizę Sentymentu (NLP):** Przetwarzanie wpisów z mediów społecznościowych (Twitter/X) przy użyciu modeli językowych (BERT).

Główną hipotezą badawczą jest sprawdzenie, czy dodanie sygnału sentymentu poprawia **kierunkową trafność prognoz (Directional Accuracy)** w porównaniu do modeli opartych wyłącznie na cenie.

---

## 📂 Struktura Projektu

```text
MultiModalFinancialPrediction/
│
├── main.py                  # Główny skrypt uruchamiający benchmark (Runner)
├── utils.py                 # Funkcje pomocnicze (pobieranie danych, metryki, wykresy)
├── requirements.txt         # Zależności projektu
│
├── models/                  # Implementacje modeli predykcyjnych
│   ├── random_walk.py       # Baseline naiwny
│   ├── arima.py             # Model statystyczny (Ceny)
│   ├── arima_stationary.py  # Model statystyczny (Zwroty %)
│   ├── lstm.py              # Sieć neuronowa (Ceny)
│   ├── lstm_stationary.py   # Sieć neuronowa (Zwroty %)
│   └── dashboard.py         # Generowanie raportów porównawczych
│
└── results/                 # (Generowane automatycznie) Wykresy i raporty
    ├── BTC-USD/
    ├── ETH-USD/
    └── ^GSPC/

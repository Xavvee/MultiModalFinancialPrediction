# Podpisy pod rysunki

Rysunki zapisane w `figures/` w dwóch formatach: `.png` (300 dpi, do edytorów tekstu) oraz `.pdf` (wektor, do LaTeX-a).

Oznaczenie RECORDED wskazuje rysunki zbudowane z zapisanych wartości, ponieważ ich ponowne przeliczenie wymaga godzin obliczeń. Pozostałe przeliczają się z danych przy każdym uruchomieniu.

## Rys. 1. (skalowanie)

Zależność zmierzonej korelacji sentymentu z ceną tego samego dnia od liczby wpisów, z których liczona jest średnia. Wzorzec powtarza się na trzech niezależnych zbiorach i dwóch klasach aktywów, co wskazuje na błąd próbkowania jako źródło tłumienia efektu przy rzadkich danych.

## Rys. 2. (kierunek_zaleznosci)

Rozdzielenie dwóch kanałów przy różnych sposobach przecięcia doby. Okna predyktora i celu nigdzie się nie nakładają. Wąsy oznaczają 95% przedziały ufności. Kanał reakcji jest silny przy każdym cięciu, kanał predykcji oscyluje wokół zera; przedziały obu kanałów nigdzie się nie stykają.

## Rys. 3. (kwantyle_luka)

Średni ruch kursu według rangi sentymentu z okna zamkniętej sesji, grupy tworzone w obrębie każdej sesji. Uporządkowanie luki otwarcia jest monotoniczne, a rozpiętość między skrajnymi grupami wynosi około 30 punktów bazowych. Ta sama procedura zastosowana do ruchu śródsesyjnego nie daje żadnego uporządkowania, co wskazuje na natychmiastową wycenę informacji.

## Rys. 4. (walidacja_poza_proba)

Jakość prognozy luki otwarcia dla obserwacji nieużytych w estymacji. W najostrzejszym wariancie — spółki nieznane modelowi w okresie nieobjętym treningiem — same cechy cenowe nie mają zdolności prognostycznej, a dodanie sentymentu daje wynik istotny. RECORDED: wartości z stocknet_oos.py.

## Rys. 5. (dogecoin_uwaga)

Nadwyżkowa zmiana kursu Dogecoina po wpisach dotyczących tej kryptowaluty względem pozostałych wpisów tego samego autora. Wzrost jest natychmiastowy i istotny, zanika w ciągu czterech godzin, a po tygodniu przechodzi w istotne odwrócenie — zgodnie z przewidywaniem teorii uwagi. Wpisy pozytywne i negatywne dają ten sam efekt, co wyklucza wydźwięk jako kanał oddziaływania. Zastrzeżenie: przy 101 zdarzeniach okna tygodniowe zachodzą na siebie.

## Rys. 6. (reddit_opoznienie)

Korelacja dziennego sentymentu z portali Reddit ze zwrotem dnia poprzedniego, bieżącego i następnego. Najsilniejszy związek zachodzi z dniem poprzednim, co odzwierciedla dłuższy, refleksyjny charakter wpisów na tej platformie. Wąsy oznaczają 95% przedziały ufności.

## Rys. 7. (per_konto)

Wynik przeszukania wszystkich kont aktywnych przez co najmniej 25 dni (54 608 testów korelacji) zestawiony z tym samym przeszukaniem na przetasowanym szeregu zwrotów. Dane prawdziwe nie produkują więcej pozornie istotnych kont niż sam przypadek, a najsilniejsza korelacja jest w obu przypadkach niemal identyczna. RECORDED: wartości z analysis/cohorts.py.

## Rys. 8. (ablacja)

Trafność kierunkowa dwóch identycznych sieci GRU różniących się wyłącznie dostępem do strumieni sentymentu, uśredniona z trzech ziaren losowych na 1560 dniach testowych. Różnica wynosi 0,28 punktu procentowego przy p = 0,86, a oba warianty pozostają poniżej trywialnej stałej predykcji (linia przerywana). RECORDED: wartości z eksperymentu ablacyjnego.

## Rys. 9. (test_placebo)

Porównanie siły zależności mierzonej w przód i wstecz dla czterech hipotez, które przy standardowym teście wyglądały na potwierdzone. W trzech przypadkach zależność wsteczna jest równie silna lub silniejsza, co wskazuje na relację współbieżną, a nie prognostyczną. Wyłącznie w przypadku luki otwarcia efekt w przód wyraźnie dominuje. Wynik dla rozproszenia opinii przeszedł wcześniej odporne błędy standardowe Newey-West oraz bootstrap blokowy — obalił go dopiero ten test. RECORDED: wartości z analysis/hypotheses.py, reddit_attention.py i stocknet_decisive.py.

## Rys. 10. (moc_statystyczna)

Zmierzone korelacje wraz z 95% przedziałami ufności, zestawione z najmniejszym efektem, jaki dana liczebność próby pozwalała wykryć przy mocy 80% (znacznik pionowy). Wyniki zerowe uzyskane na dużych próbach wykluczają efekty powyżej wskazanego progu; wynik dla kohorty wielorybów, oparty na 222 dniach, takiego wykluczenia nie umożliwia. Dwie ostatnie pozycje pokazują dla porównania, jak wygląda efekt faktycznie wykryty.


# Predikcija broja osoba u prostoriji na osnovu senzorskih podataka

Predmetni projekat iz predmeta **Softverski algoritmi u sistemima automatskog upravljanja (SAUSAU), 2026.**

## Opis problema

Cilj je proceniti **koliko se osoba nalazi u prostoriji** na osnovu očitavanja senzora, bez kamera i bez ručnog prijavljivanja. Takva procena se u praksi koristi za upravljanje grejanjem, ventilacijom i osvetljenjem — sistem može da smanji potrošnju kada je prostorija prazna, a da pojača ventilaciju kada je puna.

Problem je **višeklasna klasifikacija sa četiri klase**:

| Klasa | Značenje |
|---|---|
| 0 | prazna prostorija |
| 1 | jedna osoba |
| 2 | dve osobe |
| 3 | tri osobe |

## Dataset

**Room Occupancy Estimation** — javno dostupan skup podataka:

- https://www.kaggle.com/datasets/joebeachcapital/room-occupancy-estimation

Skup potiče iz UCI Machine Learning Repository-ja, gde je dostupan pod istim nazivom.

Sadrži **10.129 merenja** prikupljenih sa **7 senzorskih čvorova**, na svakih ~30 sekundi. Merenja su snimana tokom **7 kalendarskih dana** u dva odvojena perioda (22–26.12.2017. i 10–11.01.2018.).

Atributi: 4 senzora temperature, 4 senzora svetlosti, 4 senzora zvuka, CO₂ i njegov nagib, i 2 PIR senzora pokreta, uz `Date`, `Time` i ciljnu promenljivu `Room_Occupancy_Count`.



## Instalacija

Potreban je Python 3.10 ili noviji (razvijano na 3.14.4).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

## Pokretanje

Kompletan proces, od sirovog CSV-a do eksportovanog modela:

```bash
python pipeline.py
```

Pojedinačni korak:

```bash
python pipeline.py --korak 5
python pipeline.py --spisak      # ispis svih koraka
```

Pipeline ima 11 koraka i pokreće ih ovim redosledom:

| # | Korak | Izlaz |
|---|---|---|
| 1 | Učitavanje i pregled dataseta | `results/01_...` |
| 2 | Preprocesiranje i enkodiranje | `results/02_...`, `data/processed/occupancy_processed.csv` |
| 3 | Analiza epizoda zauzetosti | `results/03_...` |
| 4 | Finalna podela podataka i CV foldovi | `results/04_...`, `results/05_...`, `data/processed/podela.csv` |
| 5 | EDA (samo development skup) | `results/06_...`, `results/figures/` |
| 6 | Početno poređenje modela | `results/07_...` |
| 7 | Poređenje: sa vremenskim atributima i bez njih | `results/08_...` |
| 8 | Podešavanje hiperparametara | `results/09_...` |
| 9 | Odabir najznačajnijih atributa | `results/10_...`, `results/11_...` |
| 10 | Konačna evaluacija na zaključanom test skupu | `results/12_...` |
| 11 | Eksport konačnog modela | `results/13_...`, `models/finalni_model.joblib` |

Napomena o numeraciji: oznake **faza** u izvornom kodu i izveštajima prate redosled faza iz specifikacije projekta, dok brojevi **koraka** iznad prate redosled izvršavanja. Zbog toga se podela podataka (faza 5) izvršava kao korak 4, a EDA (faza 4) kao korak 5 — podela se namerno zaključava pre eksplorativne analize.

### Streamlit aplikacija

Aplikacija nije deo pipeline-a i pokreće se zasebno. Zahteva da je prethodno izvršen korak 11 (eksport modela):

```bash
streamlit run app/app.py
```

Aplikacija samo **učitava** već istrenirani model iz `models/` — ne trenira ga pri pokretanju.

## Konačni model

**Stablo odlučivanja (`DecisionTreeClassifier`)** sa hiperparametrima nađenim pomoću `GridSearchCV`:

```python
class_weight='balanced', criterion='entropy',
max_depth=3, min_samples_leaf=5, min_samples_split=2
```

Model koristi **5 od 16 senzorskih atributa**, izabranih u fazi odabira najznačajnijih atributa:

| Atribut | Opis |
|---|---|
| `S2_Light` | svetlost, čvor 2 |
| `S1_Light` | svetlost, čvor 1 |
| `S3_Light` | svetlost, čvor 3 |
| `S4_Light` | svetlost, čvor 4 |
| `S2_Temp` | temperatura, čvor 2 |

Redosled atributa je sačuvan zajedno sa modelom, pa aplikacija ne mora da ga pretpostavlja.

Skaliranje nije primenjeno jer stablo odlučivanja deli podatke po pragovima i ne zavisi od razmere atributa. Model je treniran na **development skupu od 6.608 merenja**; test skup nije korišćen u treningu.

## Rezultati na test skupu

Test skup čine **cele sesije snimanja 2 i 6** (3.518 merenja). Korišćen je **samo u fazi konačne evaluacije i nikada za razvojne odluke** — ni za EDA, ni za izbor atributa, ni za izbor modela, ni za podešavanje hiperparametara.

| Metrika | Vrednost |
|---|---|
| Accuracy | **0,8761** |
| macro F1 | **0,6491** |
| Balanced accuracy | **0,6737** |

Po klasama:

| Klasa | Precision | Recall | F1 | Uzoraka |
|---|---|---|---|---|
| 0 osoba | 1,000 | 0,954 | **0,977** | 2.912 |
| 1 osoba | 1,000 | 0,961 | **0,980** | 128 |
| 2 osobe | 0,210 | 0,421 | 0,280 | 140 |
| 3 osobe | 0,361 | 0,358 | 0,360 | 338 |

Tokom unakrsne validacije na razvojnom skupu, trivijalni model koji uvek predviđa najčešću klasu ostvario je tačnost 0,797 i macro F1 od 0,222. Rezultati konačnog modela prikazani u prethodnoj tabeli odnose se na zaseban test skup, pa se ove vrednosti ne porede direktno. Zbog neuravnoteženosti klasa, macro F1 je izabran kao glavna metrika.


## Poznata ograničenja

1. **Model dobro razlikuje 0 i 1 osobu, ali teško razlikuje 2 od 3.** Klase 0 i 1 imaju F1 preko 0,97, dok klase 2 i 3 imaju F1 od 0,28 i 0,36. Model pouzdano prepoznaje *da li* neko jeste u prostoriji, ali slabije *koliko* ih je.

2. **Prazna, topla i slabo osvetljena prostorija se ponekad klasifikuje kao tri osobe.** Ovo pravilo je nastalo zato što su epizode sa tri osobe u razvojnom skupu snimane kasno popodne i uveče.

3. **CO₂ nije među izabranim atributima, iako je kod dela test grešaka sadržao korisnu informaciju.** Kod pogrešno klasifikovanih merenja sa tri osobe medijana CO₂ bila je 1.095 ppm naspram 910 ppm kod tačno klasifikovanih. Model nije menjan zbog ovog nalaza, jer bi to značilo korišćenje test skupa za naknadni izbor atributa. Ostaje kao ideja za budući rad.

4. **Mali broj nezavisnih događaja.** U celom skupu identifikovane su 33 epizode zauzetosti, a merenja unutar epizode su jako vremenski zavisna. Sve metrike su zato procenjene sa značajnom nesigurnošću.

5. **Svetlosni atributi pokazali su značajnu prediktivnu informaciju** u razvojnom skupu i njihova korisnost održala se kroz validaciju na odvojenim sesijama. To ne znači da su univerzalno pouzdan pokazatelj broja osoba — u prostoriji sa drugačijim režimom osvetljenja rezultati mogu biti bitno drugačiji.

## Dokumentacija

Detaljan opis postupka, obrazloženja odluka i diskusija rezultata nalaze se u [DOKUMENTACIJA.md](DOKUMENTACIJA.md).

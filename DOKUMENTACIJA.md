# Predikcija broja osoba u prostoriji na osnovu senzorskih podataka

**Dokumentacija predmetnog projekta — SAUSAU 2026**

---

## 1. Opis problema i cilj

Cilj projekta je procena **broja osoba u prostoriji** isključivo na osnovu očitavanja ambijentalnih senzora. Za razliku od kamera, senzorski pristup ne narušava privatnost, jeftiniji je i ne zahteva intervenciju korisnika.

Praktična primena je u sistemima automatskog upravljanja zgradama: grejanje, ventilacija i osvetljenje mogu se prilagoditi stvarnoj zauzetosti prostorije umesto fiksnom rasporedu. Razlika između prazne i pune prostorije direktno se prevodi u uštedu energije, dok razlika između jedne i tri osobe utiče na potrebnu razmenu vazduha.

Problem je formulisan kao **višeklasna klasifikacija sa četiri klase**: 0, 1, 2 ili 3 osobe.

Metodološki cilj projekta bio je da procena performansi bude **poštena**, odnosno da odražava ponašanje modela na podacima koje nije video. Kod senzorskih vremenskih serija to nije trivijalno i predstavljalo je centralni izazov ovog projekta.

---

## 2. Opis dataseta i atributa

Korišćen je javni skup podataka **Room Occupancy Estimation**, dostupan na Kaggle platformi (https://www.kaggle.com/datasets/joebeachcapital/room-occupancy-estimation) i poreklom iz UCI Machine Learning Repository-ja.

### Osnovni podaci

| Svojstvo | Vrednost |
|---|---|
| Broj merenja | 10.129 |
| Broj kolona | 19 (17 ulaznih + `Date`/`Time`, plus ciljna promenljiva) |
| Period uzorkovanja | ~30 s (izmerena medijana: 31 s) |
| Broj senzorskih čvorova | 7 |
| Vremenski raspon | 22.12.2017. 10:49 — 11.01.2018. 09:00 |

### Atributi

| Grupa | Kolone | Jedinica |
|---|---|---|
| Temperatura | `S1_Temp` … `S4_Temp` | °C |
| Svetlost | `S1_Light` … `S4_Light` | lux |
| Zvuk | `S1_Sound` … `S4_Sound` | V (amplituda) |
| Ugljen-dioksid | `S5_CO2` | ppm |
| Nagib CO₂ | `S5_CO2_Slope` | ppm/min |
| Detekcija pokreta | `S6_PIR`, `S7_PIR` | binarno 0/1 |
| Vreme | `Date`, `Time` | tekst |
| **Cilj** | `Room_Occupancy_Count` | 0, 1, 2 ili 3 |

### Raspodela klasa

| Broj osoba | Merenja | Udeo |
|---|---|---|
| 0 | 8.228 | 81,23 % |
| 1 | 459 | 4,53 % |
| 2 | 748 | 7,38 % |
| 3 | 694 | 6,85 % |

Odnos najbrojnije i najređe klase je **17,9 : 1**. Skup je izrazito nebalansiran, što je odredilo izbor metrika kroz ceo projekat.

Napomena o opisu skupa: iako se u nekim izvorima navodi da su podaci prikupljani „tokom četiri dana", provera je pokazala **7 kalendarskih datuma u dva odvojena perioda** — 22–26. decembra 2017. i 10–11. januara 2018, sa pauzom od 366 sati između njih.

---

## 3. Početno preprocesiranje

Vodeći princip u ovoj fazi bio je: **menja se samo ono za šta postoji dokaz da treba menjati.**

### Provera kvaliteta podataka

| Provera | Rezultat | Postupak |
|---|---|---|
| Nedostajuće vrednosti | **0** | imputacija nije potrebna |
| Prazni stringovi | 0 | — |
| Potpuno identični redovi | **0** | nema šta da se ukloni |
| Vrednosti van fizičkih opsega | nema | — |
| Konstantne kolone | nema | — |
| Neuspešno parsiranih vremena | 0 | — |
| `S6_PIR`, `S7_PIR` binarni | potvrđeno `[0, 1]` | — |

**Nijedan red nije uklonjen.** Broj merenja pre i posle preprocesiranja je isti: 10.129.

### Ponovljena senzorska očitavanja

Postoji **1.301 red (12,8 %)** čija su senzorska očitavanja identična nekom drugom redu kada se zanemari vreme. Ovi redovi **nisu uklonjeni**, jer to nisu duplikati u uobičajenom smislu — reč je o prostoriji u kojoj se ništa ne menja, pa senzori vraćaju iste vrednosti. Njihovo brisanje bi uklonilo informaciju o trajanju stanja i veštački izmenilo raspodelu klasa.

### Ekstremne vrednosti

Nijedna vrednost nije van fizički mogućih opsega, pa nijedna nije uklonjena. Kod senzora ekstremna vrednost obično znači da se događaj stvarno desio — upaljeno svetlo, ulazak osobe, zvuk — a to je upravo signal koji model treba da nauči.

### Enkodiranje

| Kolona | Odluka | Obrazloženje |
|---|---|---|
| `Room_Occupancy_Count` | bez izmene | već ceo broj 0–3; `LabelEncoder` nepotreban |
| `S6_PIR`, `S7_PIR` | bez izmene | već binarni; one-hot nema smisla |
| `Date`, `Time` | **uklonjeni** | tekst; numerički kodiran datum bio bi identifikator dana eksperimenta, a ne fizička veličina |
| → izvedeno | `minut_u_danu`, `sat`, `dan_u_nedelji`, `vikend`, `minut_sin`, `minut_cos` | ciklično kodiranje sinusom i kosinusom da 23:59 i 00:00 budu susedni, a ne na suprotnim krajevima skale |

Kategorijskih kolona nema — svi senzorski atributi su numerički.

Izvedeni vremenski atributi su u ovoj fazi tretirani **isključivo kao kandidati**, a njihova korisnost je proverena eksperimentalno (poglavlje 7).

### Skaliranje

Skaliranje **nije primenjeno u fazi preprocesiranja**, i to namerno. `StandardScaler` uči srednju vrednost i standardnu devijaciju iz podataka. Da je fitovan na celom skupu pre podele, statistika test podataka bi ušla u transformaciju trening podataka — što je oblik curenja informacija.

Umesto toga, skaliranje je stavljeno unutar `sklearn.pipeline.Pipeline` i primenjeno samo tamo gde je potrebno (logistička regresija), pri čemu se skaler fituje isključivo na trening delu svakog folda. Modeli zasnovani na stablima ga ne zahtevaju jer dele podatke po pragovima, a monotone transformacije ne menjaju te podele.

---

## 4. Analiza vremenske strukture i obrazloženje podele

Ovo je metodološki najvažnije poglavlje projekta.

### Zašto običan nasumični train/test split nije primenjen

Merenja su vršena na svakih ~30 sekundi. Dva uzastopna reda opisuju gotovo isti trenutak i imaju skoro identične vrednosti svih senzora.

Kada bi se primenio standardni `train_test_split(shuffle=True)`, susedna, praktično identična merenja završila bi jedno u trening a drugo u test skupu. Model bi tada na testu prepoznavao trenutke koje je već video u treningu, a ne generalizovao na nove situacije. Rezultat bi bio **lažno visok** i metodološki neispravan. Takva podela nije sprovedena, pa se konkretna vrednost koju bi dala ne navodi.

Da bi se to izbeglo, cela podela je izgrađena na jedinicama koje su vremenski povezane.

### Sesije snimanja

Analiza razmaka između uzastopnih merenja otkrila je **7 prekida snimanja dužih od 2 minuta**, koji dele podatke na **8 sesija**:

| Sesija | Period | Merenja | Prisutne klase |
|---|---|---|---|
| 0 | 22.12. 10:49 – 11:36 | 92 | 1, 2 |
| 1 | 22.12. 11:39 – 12:42 | 124 | 1, 2, 3 |
| 2 | 22.12. 13:08 – 19:32 | 739 | **0, 1, 2, 3** |
| 3 | 22.12. 19:35 – 19:36 | 3 | 3 |
| 4 | 22.12. 19:39 – 23.12. 16:37 | 2.432 | 0, 1, 2 |
| 5 | 23.12. 16:40 – 24.12. 09:10 | 1.915 | 0, 2, 3 |
| 6 | 25.12. 09:11 – 26.12. 09:10 | 2.779 | samo 0 |
| 7 | 10.01. 15:25 – 11.01. 09:00 | 2.045 | 0, 2, 3 |

### Epizode zauzetosti

Unutar sesija podaci su podeljeni na **epizode** — neprekidne intervale u kojima je broj osoba konstantan. Identifikovane su **33 epizode zauzetosti**:

| Klasa | Broj epizoda | Merenja | Prosečno trajanje |
|---|---|---|---|
| 0 | 6 | 8.228 | 708,8 min |
| 1 | **5** | 459 | 46,9 min |
| 2 | 12 | 748 | 31,7 min |
| 3 | 10 | 694 | 35,5 min |

Merenja unutar jedne epizode su jako vremenski zavisna — ako je jedna osoba sedela u prostoriji 40 minuta, to je oko 80 redova koji opisuju isti neprekidni događaj. Zbog toga **broj epizoda, a ne broj redova, određuje koliko različitih situacija je model zaista video** za datu klasu.

Klasa 1 je najkritičnija: postoji u svega 5 epizoda, od kojih jedna traje 31 sekundu (2 merenja), pa su realno **4 upotrebljive epizode**, sve 22. i 23. decembra.

### Ispitane strategije podele

Sistematski je upoređeno sedam kandidata. Ključni nalazi:

| Varijanta | Problem |
|---|---|
| Naivna hronološka 60/20/20 po redovima | validacioni skup nema klase 1 i 2, test nema klasu 1 |
| Hronološka po sesijama | ni validacija ni test nemaju klasu 1; test 47,6 % podataka |
| Po kalendarskim danima | validacija nema klasu 1; tri epizode presečene ponoćem |
| Epizodna, sa sečenjem svih klasa | presečene epizode manjinskih klasa — najgori profil curenja |

### Usvojena podela

**Test skup = cele sesije snimanja 2 i 6.**

Obrazloženje:
- obe sesije ostaju **cele**, pa nijedna epizoda nije razdvojena između razvojnog i test skupa;
- sesija 2 je jedina sesija koja sama sadrži **sve četiri klase**;
- sesija 6 je 24 sata prazne prostorije, čime test dobija realan udeo klase 0;
- sesija 5 je namerno ostavljena u razvojnom skupu jer nosi veliku epizodu klase 3.

| Skup | Merenja | Udeo | kl. 0 | kl. 1 | kl. 2 | kl. 3 | Sesije |
|---|---|---|---|---|---|---|---|
| Development | 6.608 | 65,2 % | 5.316 | 331 | 608 | 353 | 0, 1, 4, 5, 7 |
| **Test (zaključan)** | 3.518 | 34,7 % | 2.912 | 128 | 140 | 338 | **2, 6** |
| Purge | 3 | 0,04 % | 0 | 0 | 0 | 3 | 3 |

**Broj epizoda razdvojenih između razvojnog i test skupa: 0.**

Test skup ima 82,8 % prazne prostorije, što gotovo tačno prati raspodelu celog skupa (81,2 %).

### Purge margina

Oko test sesija primenjena je **zaštitna margina od 5 minuta**: svi redovi razvojnog skupa koji su vremenski bliži od 5 minuta bilo kom test merenju izbačeni su iz oba skupa. Time se uklanja korelacija preko granice. Izbačena su svega 3 merenja (sesija 3, koja počinje 3,6 minuta nakon kraja sesije 2).

### Unakrsna validacija

Umesto fiksnog validacionog skupa korišćena je **`StratifiedGroupKFold` sa 2 folda i `sesija_id` kao grupom**:

| Fold | Validacija | Trening | kl. 0 | kl. 1 | kl. 2 | kl. 3 |
|---|---|---|---|---|---|---|
| 1 | sesije 4, 5 | sesije 0, 1, 7 | 3.565 | 254 | 398 | 130 |
| 2 | sesije 0, 1, 7 | sesije 4, 5 | 1.751 | 77 | 210 | 223 |

Oba folda sadrže sve četiri klase u upotrebljivim količinama. Broj foldova je ograničen na 2 jer klasa 3 u razvojnom skupu ima samo dve velike epizode (130 i 199 merenja), pa se više foldova ne bi moglo smisleno popuniti.

Grupisanje po sesiji je strože od grupisanja po epizodi, jer epizode iz iste sesije dele uslove istog dana. Model tako nikada ne vidi sesiju na kojoj se ocenjuje.

### Zaključavanje test skupa

Test skup je zaključan pre eksplorativne analize. Zaštita je ugrađena u kod: funkcija `ucitaj_skup()` u `src/data_split.py` odbija da učita test skup i podiže `PermissionError` osim ako se eksplicitno prosledi `dozvoli_test=True`. Taj argument se u celom projektu prosleđuje **tačno jednom** — u `src/evaluate.py`, u fazi konačne evaluacije.

Test skup nije korišćen ni za EDA odluke, ni za izbor atributa, ni za izbor modela, ni za podešavanje hiperparametara.

---

## 5. Eksplorativna analiza (EDA)

EDA je rađena **isključivo na razvojnom skupu** (6.608 merenja, 22 epizode).

### Raspodele atributa

- **Svetlost je dvomodalna, ne zvonasta.** Kod sva četiri senzora 61,6–63,8 % merenja iznosi tačno 0. Prostorija je ili mračna ili osvetljena, bez sredine.
- **Zvuk je izrazito desno asimetričan** (koeficijent asimetrije 5,6–13,9). Odnos maksimuma i medijane iznosi 49–69×: tišina uz retke kratke skokove.
- `S5_CO2_Slope` je jedini simetričan atribut (asimetrija −0,09), centriran oko nule, što je očekivano za izvod.

### Anomalije

Senzor `S2_Temp` odudara od ostala tri temperaturna senzora:

| | S1_Temp | **S2_Temp** | S3_Temp | S4_Temp |
|---|---|---|---|---|
| Maksimum | 26,31 | **29,00** | 26,06 | 26,50 |
| Opseg | 1,37 | **4,25** | 1,62 | 1,44 |
| Ekstrema po IQR pravilu | 0 | **448** | 0 | 0 |

Ostala tri senzora nemaju nijedan IQR-ekstrem, dok ih `S2_Temp` ima 448 (6,8 %). Verovatno objašnjenje je da je taj čvor bliže izvoru toplote ili osobama. Vrednost od 29 °C je fizički moguća za zatvorenu prostoriju, pa atribut **nije menjan ni uklanjan**.

### Korelacije unutar grupa senzora

| Grupa | Prosečno *r* | Opseg |
|---|---|---|
| Temperatura | **0,844** | 0,760 – 0,968 |
| Svetlost | 0,661 | 0,522 – 0,826 |
| Zvuk | 0,546 | 0,416 – 0,701 |

Temperaturni senzori su najredundantniji, zvučni najmanje — što ima fizičko objašnjenje: toplota se razliva po prostoriji, a zvuk je lokalan.

Iznad praga |r| ≥ 0,90 nalaze se samo dva para: `S1_Temp`–`S3_Temp` (0,968) i `S1_Temp`–`S4_Temp` (0,913).

Postoji i jaka međugrupna korelacija: `S1_Temp`–`S5_CO2` iznosi 0,851, jer i temperatura i CO₂ rastu sa prisustvom ljudi.

### CO₂ i nagib CO₂

Korelacija između `S5_CO2` i `S5_CO2_Slope` iznosi **−0,117** — nisu redundantni. Nivo govori koliko se CO₂ nakupilo, a nagib da li se trenutno nakuplja ili opada.

U razvojnom skupu medijane oba atributa **monotono rastu kroz klase**:

| Klasa | Medijana CO₂ | Medijana nagiba |
|---|---|---|
| 0 | 360 | 0,000 |
| 1 | 415 | 0,300 |
| 2 | 645 | 1,087 |
| 3 | 705 | 2,196 |

Ovo je opis ovog skupa podataka, a ne tvrdnja o ponašanju senzora uopšte.

### PIR senzori

| Klasa | S6 aktivan | S7 aktivan | Bar jedan |
|---|---|---|---|
| 0 | 0,2 % | 0,2 % | 0,2 % |
| 1 | 36,9 % | 4,5 % | 37,8 % |
| 2 | 48,8 % | 36,3 % | 59,7 % |
| 3 | 47,9 % | 62,9 % | 73,1 % |

U samo **0,2 %** merenja prazne prostorije bio je aktivan bar jedan PIR senzor — dakle PIR pokazuje veoma visoku specifičnost za praznu prostoriju, lažna detekcija je retka.

Obrnuto, u **42,3 %** merenja sa bar jednom osobom oba PIR senzora bila su mirna. PIR reaguje na pokret, ne na prisustvo, pa osoba koja mirno sedi ostaje nezapažena.

### Zapažanje o svetlosti

Medijane svetlosti **nisu monotone** po broju osoba:

| Atribut | kl. 0 | kl. 1 | kl. 2 | **kl. 3** |
|---|---|---|---|---|
| `S1_Light` | 0 | 122 | 146 | **10** |
| `S2_Light` | 0 | 35 | 231 | **13** |
| `S4_Light` | 0 | 44 | 20 | **16** |

Kod više senzora klasa 3 ima nižu medijanu svetlosti od klasa 1 i 2. U EDA fazi je postavljena **hipoteza** da su epizode klase 3 u razvojnom skupu snimane kasno popodne i uveče, pa je izmerena svetlost povezana i sa sesijom i dobom dana, a ne samo sa brojem osoba.

Hipoteza je proverena kasnije (poglavlja 7 i 9) i **nije potvrđena** — svetlost se pokazala kao nosilac stvarne prediktivne informacije.

---

## 6. Početno poređenje modela

Trenirane su osnovne verzije četiri modela iz specifikacije, bez podešavanja hiperparametara, na 16 senzorskih atributa uz usvojeni 2-fold CV. Dodat je i `DummyClassifier` kao baseline.

Prikazane su dve početne postavke — podrazumevana i `class_weight='balanced'` — jer pri udelu klase 0 od 80 % podrazumevani model može uopšte ne predviđati retke klase. Kod XGBoost-a, koji nema parametar `class_weight`, korišćeno je ekvivalentno ponderisanje uzoraka.

| Model | Postavka | F1 fold 1 | F1 fold 2 | macro F1 | Bal. acc | Accuracy |
|---|---|---|---|---|---|---|
| RandomForest | balanced | 0,592 | 0,708 | **0,650** | 0,669 | 0,876 |
| DecisionTree | balanced | 0,538 | 0,728 | 0,633 | 0,691 | 0,882 |
| RandomForest | podrazumevano | 0,539 | 0,679 | 0,609 | 0,672 | 0,847 |
| LogisticRegression | podrazumevano | 0,519 | 0,673 | 0,596 | 0,582 | 0,850 |
| DecisionTree | podrazumevano | 0,581 | 0,601 | 0,591 | 0,705 | 0,843 |
| LogisticRegression | balanced | 0,500 | 0,671 | 0,586 | 0,605 | 0,843 |
| XGBoost | podrazumevano | 0,520 | 0,651 | 0,585 | 0,616 | 0,859 |
| XGBoost | balanced | 0,479 | 0,612 | 0,545 | 0,567 | 0,837 |
| *Baseline (najčešća klasa)* | — | 0,225 | 0,218 | *0,222* | *0,250* | *0,797* |

### Zašto tačnost nije glavna metrika

Baseline koji uvek predviđa „prazna prostorija" postiže **79,7 % tačnosti**. Najbolji model postiže 87,6 % — razlika od svega 8 procentnih poena. Ali macro F1 raste sa **0,222 na 0,650**, skoro trostruko.

Da je praćena samo tačnost, zaključak bi bio da modeli jedva nešto donose. Zato je kao glavna metrika usvojen **macro F1**, koji svakoj klasi daje istu težinu, uz balanced accuracy i metrike po klasama.

### Ostala zapažanja

- Razlika između foldova je velika (0,10–0,19 macro F1). Foldovi validiraju različite sesije, pa to ukazuje da modeli slabo generalizuju na nov dan — što je realan scenario primene.
- `class_weight='balanced'` pomaže stablima, ali odmaže logističkoj regresiji i XGBoost-u. Zbog toga ponderisanje nije globalno fiksirano, već je uvršteno u prostor pretrage hiperparametara.

---

## 7. Analiza vremenskih atributa

Specifikacija projekta zahteva proveru da li izvedeni vremenski atributi zaista poboljšavaju generalizaciju ili samo kodiraju raspored eksperimenta.

Upoređena su dva skupa pod identičnim uslovima: 16 senzorskih atributa naspram 16 senzorskih + 6 vremenskih (`minut_u_danu`, `sat`, `dan_u_nedelji`, `vikend`, `minut_sin`, `minut_cos`).

Ključna provera nije prosek, već **slaganje foldova**: foldovi validiraju različite sesije, pa ako vremenski atributi pomažu samo u jednom foldu, to je znak da model uči raspored a ne fiziku.

| Model | Postavka | Δ fold 1 | Δ fold 2 | Δ prosek | Dosledno? |
|---|---|---|---|---|---|
| LogisticRegression | podrazumevano | −0,029 | 0,000 | −0,015 | ne |
| DecisionTree | podrazumevano | −0,058 | −0,201 | −0,130 | oba lošija |
| RandomForest | podrazumevano | −0,153 | −0,063 | −0,108 | oba lošija |
| XGBoost | podrazumevano | −0,038 | −0,062 | −0,050 | oba lošija |
| LogisticRegression | balanced | −0,174 | 0,000 | −0,087 | ne |
| **DecisionTree** | **balanced** | **+0,221** | **−0,224** | −0,001 | **suprotno** |
| RandomForest | balanced | −0,177 | −0,038 | −0,107 | oba lošija |
| XGBoost | balanced | −0,065 | −0,019 | −0,042 | oba lošija |

**Vremenski atributi podigli su prosek u 0 od 8 kombinacija i pomogli u oba folda u 0 od 8.**

Najrečitiji je red za DecisionTree sa `balanced`: **+0,221 u prvom foldu i −0,224 u drugom.** Vremenski atributi tu ne poboljšavaju predikciju — oni pamte kada se šta dešavalo u sesijama 4 i 5, a to znanje je štetno na sesijama 0, 1 i 7. To je upravo obrazac koji ukazuje na učenje rasporeda eksperimenta.

Najviše strada klasa 1: kod RandomForest-a sa `balanced` F1 pada sa 0,707 na 0,304.

**Odluka: vremenski atributi se ne koriste u daljem modeliranju.** Zadržano je 16 senzorskih atributa.

---

## 8. Podešavanje hiperparametara

Pretraga je izvršena pomoću `GridSearchCV` (logistička regresija, stablo odlučivanja, Random Forest) i `RandomizedSearchCV` sa 25 kombinacija (XGBoost), uz usvojenu 2-fold podelu po `sesija_id` i kriterijum macro F1. Prostori pretrage su namerno umereni jer je skup mali.

`class_weight` (`None` i `'balanced'`) uvršten je u prostor pretrage kod modela koji ga podržavaju. Kod XGBoost-a, koji taj parametar nema, obični trening i trening sa ponderisanjem uzoraka vođeni su kao dve odvojene pretrage.

### Rezultati

| Model | F1 fold 1 | F1 fold 2 | macro F1 | Bal. acc | Accuracy | Kombinacija |
|---|---|---|---|---|---|---|
| **DecisionTree** | 0,679 | 0,703 | **0,691** | **0,758** | 0,884 | 144 |
| RandomForest | 0,594 | 0,711 | 0,653 | 0,690 | 0,876 | 64 |
| XGBoost (bez ponderisanja) | 0,573 | 0,701 | 0,637 | 0,679 | 0,867 | 25 |
| XGBoost (sample_weight) | 0,497 | 0,775 | 0,636 | 0,647 | 0,869 | 25 |
| LogisticRegression | 0,561 | 0,676 | 0,619 | 0,639 | 0,859 | 10 |

### Najbolji parametri

| Model | Parametri |
|---|---|
| **DecisionTree** | `class_weight='balanced'`, `criterion='entropy'`, `max_depth=3`, `min_samples_leaf=5`, `min_samples_split=2` |
| RandomForest | `class_weight='balanced'`, `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=1`, `min_samples_split=10`, `n_estimators=100` |
| XGBoost | `colsample_bytree=1.0`, `learning_rate=0.05`, `max_depth=7`, `n_estimators=100`, `subsample=1.0` |
| LogisticRegression | `C=10`, `class_weight=None` |

### Diskusija

**Optimalna vrednost `class_weight` nije ista za sve modele.** Stablo odlučivanja i Random Forest biraju `'balanced'`, dok logistička regresija bira `None`. Da je ponderisanje globalno fiksirano, logistička regresija bi bila nepotrebno hendikepirana. Ovo potvrđuje ispravnost odluke da se `class_weight` tretira kao hiperparametar.

**Najbolji model je stablo dubine 3.** Stablo sa svega tri nivoa nadmašuje Random Forest sa 100 stabala i XGBoost. Nije samo najbolje, nego i **najstabilnije**:

| Model | Razlika između foldova |
|---|---|
| **DecisionTree** | **0,024** |
| RandomForest | 0,118 |
| XGBoost (bez ponderisanja) | 0,128 |
| XGBoost (sample_weight) | 0,278 |

Objašnjenje je u strukturi podataka: razvojni skup ima svega 22 epizode. Složeniji modeli uče specifičnosti pojedinačne sesije i onda padnu na drugoj, dok plitko stablo to ne može pa bolje generalizuje.

Podešavanje je podiglo najbolji macro F1 sa 0,650 na **0,691**, a F1 za klasu 3 sa 0,26 na 0,36.

---

## 9. Odabir najznačajnijih atributa

### Metod

Značaj atributa računat je **unutar svakog folda**: impurity značaj na trening delu, permutaciona provera na validacionom delu istog folda. Test skup nije korišćen.

Podešeno stablo ima `max_depth=3` i može koristiti najviše 7 atributa — u praksi je koristilo 6, dok je ostalih 10 dobilo značaj tačno 0. Zbog toga stablo ne može da rangira svih 16 atributa. Za potpun poredak korišćen je **Random Forest impurity značaj**, a mere iz stabla i permutaciona provera služile su kao kontrola slaganja (Spearman 0,617 odnosno 0,343).

### Rangiranje

| # | Atribut | dt_impurity | dt_permutacija | rf_impurity |
|---|---|---|---|---|
| 1 | **S2_Light** | 0,223 | 0,237 | 0,197 |
| 2 | **S1_Light** | 0,410 | 0,362 | 0,170 |
| 3 | **S3_Light** | 0,249 | 0,103 | 0,129 |
| 4 | **S4_Light** | 0,004 | −0,014 | 0,095 |
| 5 | S2_Temp | 0 | 0 | 0,086 |
| 6 | S3_Temp | 0 | 0 | 0,076 |
| 7 | S5_CO2 | 0 | 0 | 0,075 |
| 8 | S1_Temp | 0 | 0 | 0,073 |
| 9 | S5_CO2_Slope | 0,108 | −0,005 | 0,028 |
| 10 | S4_Temp | 0 | 0 | 0,024 |
| 11–14 | zvučni senzori | ≈ 0 | ≈ 0 | 0,005 – 0,018 |
| 15–16 | S7_PIR, S6_PIR | 0 | 0 | 0,0008 / 0,0001 |

Sva četiri svetlosna senzora zauzimaju prva četiri mesta. Udeo svetlosti u ukupnom značaju iznosi **59,1 %** kod Random Forest-a i **88,5 %** kod stabla, naspram 25 % koliko bi činila da su svi atributi jednako značajni.

### Izbor broja atributa

Za svako *k* od 1 do 16 trenirano je podešeno stablo na top-*k* atributa. Kriva je nestabilna — dodavanje jednog atributa ume da obori macro F1 za 0,15 — što je posledica plitkog stabla kod kojeg jedan nov atribut menja izbor podela.

Pravilo **1 standardne greške** je izračunato i prikazano, ali **nije korišćeno kao kriterijum**: sa samo dva folda standardna greška se procenjuje iz dve vrednosti (dobijeno 0,0016), što je nepouzdano.

Umesto toga primenjen je kriterijum: najmanje *k* čiji je prosečan macro F1 unutar 0,02 od najboljeg (k = 7, 0,7118), uz zahtev da razlika između foldova ne bude bitno gora. Rezultat je **k = 5**, koja je ujedno i najstabilnija tačka na krivoj (razlika foldova 0,0085).

### Poređenje svih 16 i odabranih 5

| | Atributa | F1 fold 1 | F1 fold 2 | macro F1 | Bal. acc | Accuracy |
|---|---|---|---|---|---|---|
| Svi | 16 | 0,679 | 0,703 | 0,691 | **0,758** | 0,884 |
| **Odabrani** | **5** | 0,712 | 0,703 | **0,707** | 0,742 | **0,890** |

Podskup od 5 atributa **zadržava, čak i blago poboljšava** performanse (+0,0161 macro F1). Po klasama, F1 za klasu 1 raste sa 0,728 na 0,795, za klasu 3 sa 0,363 na 0,390, dok klasa 2 ostaje nepromenjena.

**Metodološka napomena.** Rangiranje atributa i procena odabranog podskupa rađeni su na istoj 2-fold cross-validation podeli development skupa, zbog čega razvojna CV procena za odabrani podskup može biti blago optimistična. Finalna procena modela nije pogođena ovim, jer test skup nije korišćen ni za rangiranje atributa ni za izbor njihovog broja, već isključivo u konačnoj evaluaciji.

### Dijagnostička provera: da li je svetlost informacija ili prečica?

Pošto su odabrani atributi praktično „sva svetlost plus jedna temperatura", provera hipoteze iz EDA bila je neophodna. Isti podešeni model, pod identičnim uslovima, upoređen je na tri skupa — menjan je isključivo skup atributa, bez ponovnog podešavanja hiperparametara.

| Skup | Atributa | F1 fold 1 | F1 fold 2 | macro F1 | Razlika foldova |
|---|---|---|---|---|---|
| A) svi senzori | 16 | 0,679 | 0,703 | 0,691 | 0,024 |
| **B) odabrani** | **5** | 0,712 | 0,703 | **0,707** | **0,0085** |
| C) bez svetlosti | 12 | 0,400 | 0,427 | **0,414** | 0,027 |

F1 po klasama:

| Klasa | A) svi 16 | B) odabrani 5 | C) bez svetlosti |
|---|---|---|---|
| 0 | 0,960 | 0,957 | **0,981** |
| 1 | 0,728 | **0,795** | **0,196** |
| 2 | 0,646 | 0,646 | **0,313** |
| 3 | 0,363 | **0,390** | 0,282 |

**Uklanjanje svetlosti obara macro F1 za 0,294.** Bez svetlosti model je bolji u prepoznavanju prazne prostorije (0,981), ali gubi sposobnost brojanja — klasa 1 pada sa 0,795 na 0,196.

Hipoteza iz EDA **nije potvrđena**. Argument je metodološki jak: unakrsna validacija je na nivou sesija, pa da je svetlost samo identifikator sesije, ne bi se prenela na nove sesije i rezultat bi pao. Umesto toga održala je 0,71 na neviđenim sesijama.

Formulacija koja se zadržava: **svetlosni atributi nose značajnu prediktivnu informaciju u razvojnom skupu i njihova korisnost održala se kroz validaciju na odvojenim sesijama.** To ne dokazuje da su univerzalno pouzdan pokazatelj broja osoba — sve razvojne sesije dele isti protokol eksperimenta.

Značajno je i da klasa 3 **ne nestaje** bez svetlosti (0,282 naspram 0,390), što znači da je CO₂, nagib i temperatura delimično nose.

---

## 10. Konačni model

Odluka je doneta **isključivo na osnovu razvojnog skupa, pre gledanja test rezultata.**

| Svojstvo | Vrednost |
|---|---|
| Algoritam | `DecisionTreeClassifier` |
| Hiperparametri | `class_weight='balanced'`, `criterion='entropy'`, `max_depth=3`, `min_samples_leaf=5`, `min_samples_split=2` |
| Atributi (5) | `S2_Light`, `S1_Light`, `S3_Light`, `S4_Light`, `S2_Temp` |
| Skaliranje | nije primenjeno — stablo ga ne zahteva |
| Trening skup | ceo razvojni skup, 6.608 merenja |

### Struktura stabla

```
S1_Light <= 92.00
├── S2_Temp <= 25.84
│   ├── S3_Light <= 51.50  →  0 osoba
│   └── S3_Light >  51.50  →  0 osoba
└── S2_Temp >  25.84
    ├── S1_Light <= 1.00   →  0 osoba
    └── S1_Light >  1.00   →  3 osobe
S1_Light > 92.00
├── S3_Light <= 76.50
│   ├── S2_Light <= 111.50 →  1 osoba
│   └── S2_Light >  111.50 →  2 osobe
└── S3_Light > 76.50
    ├── S3_Light <= 181.50 →  2 osobe
    └── S3_Light >  181.50 →  3 osobe
```

Prednost ovakvog modela je potpuna interpretabilnost — svako pravilo se može pročitati i proveriti, što je za odbranu projekta i za razumevanje ograničenja vrednije od nekoliko stotinki macro F1.

---

## 11. Finalna evaluacija na zaključanom test skupu

Model je treniran na celom razvojnom skupu, a zatim je **jednom** izvršena predikcija nad test skupom (sesije 2 i 6, 3.518 merenja). Nakon gledanja test rezultata **ništa nije menjano** — ni model, ni atributi, ni hiperparametri, ni podela.

### Rezultati

| Metrika | Vrednost |
|---|---|
| **Accuracy** | **0,8761** |
| **macro F1** | **0,6491** |
| **Balanced accuracy** | **0,6737** |
| Macro recall | 0,6737 |

Balanced accuracy i macro recall su za višeklasni slučaj ista veličina — prosek odziva po klasama. Navedene su obe jer ih specifikacija traži pod oba imena.

### Metrike po klasama

| Klasa | Precision | Recall | F1 | Uzoraka |
|---|---|---|---|---|
| 0 osoba | **1,000** | 0,954 | **0,977** | 2.912 |
| 1 osoba | **1,000** | 0,961 | **0,980** | 128 |
| 2 osobe | 0,210 | 0,421 | 0,280 | 140 |
| 3 osobe | 0,361 | 0,358 | 0,360 | 338 |

### Matrica konfuzije

|  | pred. 0 | pred. 1 | pred. 2 | pred. 3 |
|---|---|---|---|---|
| **stvarno 0** | **2.779** | 0 | 0 | 133 |
| **stvarno 1** | 0 | **123** | 5 | 0 |
| **stvarno 2** | 0 | 0 | **59** | 81 |
| **stvarno 3** | 0 | 0 | 217 | **121** |

### Diskusija

Model radi dve stvari veoma dobro i jednu loše.

**Prazna prostorija i jedna osoba prepoznaju se skoro savršeno.** Preciznost je 1,000 za obe klase. Nijedna osoba nikada nije proglašena praznom prostorijom, i nijedna prazna prostorija nikada nije proglašena kao jedna osoba. To znači da je model upotrebljiv za osnovnu primenu — odluku da li prostorija jeste ili nije zauzeta.

**Dve i tri osobe model ne razlikuje pouzdano.** Od 338 merenja sa tri osobe, 217 (64,2 %) proglašeno je kao dve, a od 140 merenja sa dve osobe, 81 (57,9 %) kao tri. Zamena je gotovo simetrična: model prepoznaje da je prisutna grupa ljudi, ali ne broji koliko ih je.

### Poređenje sa razvojnim CV rezultatom

| | Development CV | Test | Razlika |
|---|---|---|---|
| macro F1 | 0,7074 | 0,6491 | −0,058 |
| Balanced accuracy | 0,7420 | 0,6737 | −0,068 |
| Accuracy | 0,8903 | 0,8761 | −0,014 |

Pad je umeren i ne ukazuje na ozbiljno preprilagođavanje. Struktura se, međutim, promenila: klasa 1 je na testu **bolja** (F1 0,795 → 0,980), dok je klasa 2 znatno **lošija** (0,646 → 0,280).

Pri tumačenju ovih razlika treba imati u vidu da test sadrži samo jednu epizodu klase 1 i tri epizode klase 2, pa su obe brojke procene iz malog broja nezavisnih događaja. Takođe, precision zavisi od raspodele klasa u skupu i ne prenosi se direktno između skupova različite raspodele; recall po klasi ne zavisi direktno od prevalence, ali zavisi od toga koliko su uzorci te klase u test skupu reprezentativni.

---

## 12. Analiza grešaka i ograničenja

### Raspored grešaka

Od ukupno 3.518 test merenja, model greši na **436 (12,4 %)**.

**Sve greške su u sesiji 2. Sesija 6 — 2.779 merenja, 24 sata prazne prostorije — nema nijednu grešku.**

| Epizoda | Stvarna klasa | Merenja | Grešaka | Stopa |
|---|---|---|---|---|
| 17 | 3 | 227 | 217 | 95,6 % |
| 14 | 0 | 133 | 133 | 100 % |
| 16 | 2 | 121 | 62 | 51,2 % |
| 10 | 2 | 17 | 17 | 100 % |
| 15 | 1 | 128 | 5 | 3,9 % |
| 12 | 2 | 2 | 2 | 100 % |

Greške su **koncentrisane u nekoliko čitavih ili gotovo čitavih epizoda**, a ne rasute po test skupu. To pokazuje vremensku zavisnost uzoraka: kada model pogreši na jednom delu epizode, po pravilu greši i na ostatku iste epizode, jer su susedna merenja gotovo identična. Zbog toga ukupan broj pogrešnih redova precenjuje koliko je različitih situacija model zaista promašio.

### Najčešće zamene

| Stvarna | Predviđena | Broj | Udeo klase |
|---|---|---|---|
| 3 | 2 | 217 | 64,2 % |
| 0 | 3 | 133 | 4,6 % |
| 2 | 3 | 81 | 57,9 % |
| 1 | 2 | 5 | 3,9 % |

**Epizoda 14** je prazna prostorija koju je model u 100 % slučajeva proglasio kao tri osobe. Odgovorna je grana `S1_Light ≤ 92 → S2_Temp > 25,84 → S1_Light > 1 → klasa 3`, koja je nastala zato što su epizode sa tri osobe u razvojnom skupu snimane po mraku. To je najozbiljnija sistematska greška modela.

### Nalaz o CO₂

Senzorski profil najčešće zamene (3 predviđeno kao 2), medijane:

| Atribut | Pogrešno (217) | Tačno (121) |
|---|---|---|
| `S4_Light` | 10 | 70 |
| `S3_Light` | 180 | 272 |
| *`S5_CO2`* | ***1.095*** | *910* |
| *`S5_CO2_Slope`* | *2,20* | *1,51* |

Kod pogrešno klasifikovanih merenja **CO₂ je bio viši** (1.095 naspram 910 ppm), dakle ukazivao je na *više* osoba, a ne manje. Informacija koja bi ispravila grešku postojala je u podacima, ali je model ne koristi jer `S5_CO2` nije među izabranih pet atributa.

**Model nije menjan zbog ovog nalaza.** Dodavanje CO₂ posle uvida u test rezultate značilo bi korišćenje test skupa za naknadni izbor atributa, čime bi prijavljene metrike prestale da budu poštena procena. Nalaz ostaje kao ograničenje i kao ideja za budući rad.

### Ograničenja modela

1. **Model razlikuje prisustvo, ali ne pouzdano broji.** Klase 0 i 1 imaju F1 preko 0,97; klase 2 i 3 imaju 0,28 i 0,36.

2. **Prazna, topla i slabo osvetljena prostorija se klasifikuje kao tri osobe.** Posledica je rasporeda eksperimenta u razvojnom skupu.

3. **Mali broj nezavisnih događaja.** Ceo skup sadrži 33 epizode zauzetosti, a razvojni skup 22. Merenja unutar epizode su jako vremenski zavisna, pa su sve metrike procenjene sa značajnom nesigurnošću. Test sadrži svega jednu epizodu klase 1.

4. **Ograničena spoljašnja validnost.** Svi podaci potiču iz jedne prostorije, sa jednim rasporedom senzora i jednim protokolom eksperimenta. Model treniran ovde ne mora raditi u drugoj prostoriji bez ponovnog treniranja.

5. **Oslanjanje na svetlost.** Svetlosni atributi nose značajnu prediktivnu informaciju u razvojnom skupu i ta korisnost se održala kroz validaciju na odvojenim sesijama. To ne znači da su univerzalno pouzdan pokazatelj broja osoba — u prostoriji sa automatskim ili drugačije korišćenim osvetljenjem rezultati mogu biti bitno drugačiji.

### Mogući pravci daljeg rada

- Uključivanje CO₂ i nagiba CO₂ u model i evaluacija na **novom, nezavisnom** skupu podataka.
- Prikupljanje epizoda sa tri osobe pri različitim uslovima osvetljenja, čime bi se razbila veza između broja osoba i doba dana.
- Korišćenje vremenskih prozora umesto pojedinačnih merenja, jer trend CO₂ kroz nekoliko minuta nosi više informacije od trenutne vrednosti.

---

## 13. Deployment

### Eksport modela

Konačni model je istreniran na celom razvojnom skupu (6.608 merenja) i sačuvan pomoću `joblib` u `models/finalni_model.joblib`.

Model **nije** ponovo treniran na test skupu ni na svih 10.129 merenja, kako bi eksportovani model bio tačno onaj na koji se odnose prijavljeni finalni rezultati.

Uz model je sačuvano sve što je potrebno za reproducibilnu predikciju:

| Ključ | Sadržaj |
|---|---|
| `model` | istrenirani `DecisionTreeClassifier` |
| `atributi` | tačan **redosled** pet ulaznih atributa |
| `klase` | `[0, 1, 2, 3]` |
| `hiperparametri` | korišćena konfiguracija |
| `opsezi` | min/max/medijana po atributu iz trening podataka |
| `metapodaci` | tip modela, skup za trening, broj merenja, napomena o skaliranju, datum eksporta |

Čuvanje redosleda atributa je bitno. Model pamti imena kolona iz treninga (`feature_names_in_`) i proverava ih kada mu se prosledi `DataFrame` — pogrešan redosled tada izaziva grešku. Ali ako mu se prosledi običan niz brojeva, može se osloniti samo na pozicije i tiho bi prihvatio pogrešan redosled. Zato se redosled čuva uz model, a aplikacija uvek šalje `DataFrame` sa imenima kolona. Osam automatskih provera potvrđuje da se model učitava, da daje identične predikcije kao pre snimanja, i da odbija `DataFrame` sa pogrešnim redosledom kolona.

### Streamlit aplikacija

Aplikacija `app/app.py` omogućava korišćenje modela kroz jednostavan korisnički interfejs:

- unos vrednosti za tačno pet atributa koje model koristi, sa granicama polja preuzetim iz sačuvanih opsega;
- dugme za predikciju;
- rezultat u obliku **„Predviđeni broj osoba u prostoriji: X"**.

Uz predviđenu klasu prikazuju se i vrednosti koje vraća `predict_proba` — to su **verovatnoće koje model dodeljuje svakoj klasi**, a ne kalibrisana mera sigurnosti. Kod plitkog stabla one odgovaraju raspodeli klasa u listu u koji je uzorak pao i treba ih čitati kao orijentacioni skor.

Aplikacija **učitava** eksportovani model i ne trenira ga pri pokretanju. Redosled atributa se čita iz sačuvanog paketa, nikada se ne pretpostavlja ručno, a ulaz se prosleđuje kao `DataFrame` sa imenima kolona, pa `scikit-learn` sam odbija poziv ako se redosled ne poklapa sa treningom.

Pokretanje:

```bash
streamlit run app/app.py
```

### Reproducibilnost

Ceo proces se pokreće jednom komandom:

```bash
python pipeline.py
```

Pipeline izvršava 11 koraka istim redosledom kojim je projekat rađen — od učitavanja sirovog CSV-a do eksporta modela. Podela podataka je determinističa i snimljena u `data/processed/podela.csv`, pa se svi prijavljeni brojevi mogu reprodukovati.

---

## 14. Zaključak

Projekat je pokazao da je moguće proceniti zauzetost prostorije iz ambijentalnih senzora, ali sa jasno ograničenom preciznošću.

**Šta model radi dobro.** Razlikovanje prazne prostorije od zauzete radi gotovo bez greške: F1 od 0,977 za praznu prostoriju i 0,980 za jednu osobu, uz preciznost 1,000 za obe klase na test skupu. Za primenu u upravljanju grejanjem i ventilacijom, gde je ključna razlika između prazno i zauzeto, ovo je upotrebljiv rezultat.

**Šta model radi loše.** Razlikovanje dve od tri osobe ne funkcioniše — zamena je gotovo simetrična i model u suštini prepoznaje „grupa ljudi" bez broja. Ukupan macro F1 od 0,6491 je posledica upravo te dve slabe klase.

**Metodološki doprinos.** Najveći deo rada otišao je na to da procena bude poštena. Podaci su vremenska serija sa merenjima na 30 sekundi, pa bi standardni nasumični `train_test_split` doveo do toga da praktično identična susedna merenja završe u trening i test skupu — i do lažno visokog rezultata. Umesto toga:

- identifikovane su 33 epizode zauzetosti i 8 sesija snimanja;
- test skup čine **cele sesije 2 i 6**, tako da nijedna epizoda nije razdvojena;
- oko granice test skupa primenjena je zaštitna margina od 5 minuta;
- unakrsna validacija koristi `sesija_id` kao grupu, pa model nikad ne vidi sesiju na kojoj se ocenjuje;
- test skup je zaključan u kodu i korišćen samo u fazi konačne evaluacije, nikada za razvojne odluke.

Cena ove strogosti je niži prijavljeni rezultat nego što bi dao nasumični split, ali je taj rezultat verodostojan.

**Neočekivani nalazi.** Tri rezultata su bila suprotna očekivanjima:

1. **Stablo dubine 3 nadmašilo je Random Forest i XGBoost** i bilo najstabilnije između foldova. Uz svega 22 epizode u razvojnom skupu, složeniji modeli uče specifičnosti pojedine sesije umesto opšteg obrasca.

2. **Izvedeni vremenski atributi nisu pomogli ni u jednoj od 8 testiranih kombinacija.** Kod jedne kombinacije poboljšali su prvi fold za +0,221 a pogoršali drugi za −0,224 — jasna slika učenja rasporeda eksperimenta umesto fizičke veze.

3. **Pet atributa dalo je bolji rezultat od svih šesnaest** (macro F1 0,707 naspram 0,691). Za fizički sistem to znači da bi se, bar u ovakvom rasporedu, sistem mogao pojednostaviti sa sedam senzorskih čvorova na manji broj uz prihvatljiv gubitak.

**Otvorena pitanja.** Analiza grešaka je pokazala da je CO₂ kod dela pogrešnih predikcija sadržao informaciju koja bi grešku ispravila, ali model nije menjan jer bi to značilo korišćenje test skupa za naknadni izbor atributa. Provera te ideje zahteva nov, nezavisan skup podataka. Takođe, veza između broja osoba i doba dana u ovom skupu ostaje ograničenje spoljašnje validnosti koje bi se rešilo prikupljanjem epizoda pri različitim uslovima osvetljenja.

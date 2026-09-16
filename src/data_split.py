"""
FAZA 5 - Podela podataka na train / validation / test.

Radi se PRE svake target-aware analize. Razlog: svaka odluka doneta gledanjem
odnosa atributa i targeta na celom skupu vec je oblik curenja informacija.
Zato prvo zakljucavamo podelu, pa tek onda analiziramo - i to na trening delu.

U ovom modulu se koriste samo STRUKTURNE informacije:
gde se koja klasa nalazi u vremenu, koliko traju epizode, kako se lome sesije.
Nigde se ne gleda odnos senzorskih vrednosti i targeta.

Pokretanje:
    python src/data_split.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from config import (
    RESULTS_DIR,
    TARGET,
    BLOCK_MINUTES,
    TEST_SESIJE,
    PURGE_MINUTA,
    CV_FOLDOVA,
    SPLIT_CSV,
    MIN_REDOVA_ZA_EVALUACIJU,
    ensure_dirs,
    rel,
)
from data_preparation import load_processed, Izvestaj

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


# ====================================================================
# Struktura targeta kroz vreme
# ====================================================================
def nadji_epizode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deli podatke na EPIZODE - neprekidne intervale u kojima je broj osoba
    konstantan i nema prekida snimanja.

    Zasto je epizoda prava jedinica podele:
      Merenja unutar jedne epizode su jako vremenski zavisna. Ako je troje
      ljudi sedelo u sobi 40 minuta, to je ~80 redova koji opisuju isti
      neprekidni dogadjaj i medjusobno su vrlo slicni. Zato epizoda mora
      ostati cela u jednom skupu.
    """
    d = df.sort_values("Timestamp").reset_index(drop=True).copy()

    promena_klase = d[TARGET] != d[TARGET].shift()
    promena_sesije = d["sesija_id"] != d["sesija_id"].shift()
    d["epizoda_id"] = (promena_klase | promena_sesije).cumsum()

    epizode = d.groupby("epizoda_id").agg(
        klasa=(TARGET, "first"),
        sesija=("sesija_id", "first"),
        datum=("datum", "first"),
        pocetak=("Timestamp", "min"),
        kraj=("Timestamp", "max"),
        redova=("Timestamp", "size"),
    )
    epizode["trajanje_min"] = (
        (epizode["kraj"] - epizode["pocetak"]).dt.total_seconds() / 60
    ).round(1)
    return epizode


def blokovi_po_velicini(df: pd.DataFrame, minuti: int) -> pd.Series:
    """
    Za datu duzinu bloka vraca koliko blokova sadrzi bar jedan uzorak
    svake klase. Sluzi za izbor BLOCK_MINUTES.
    """
    d = df.sort_values("Timestamp").copy()
    pocetak = d.groupby("sesija_id")["Timestamp"].transform("min")
    protekli = (d["Timestamp"] - pocetak).dt.total_seconds() / 60
    blok = d["sesija_id"].astype(str) + "_" + (protekli // minuti).astype(int).astype(str)
    d = d.assign(_blok=blok)
    return d.groupby(TARGET)["_blok"].nunique()


def analiza_strukture() -> pd.DataFrame:
    """Strukturna analiza koja prethodi podeli. Ne dira odnos feature-target."""
    ensure_dirs()
    r = Izvestaj()

    df = load_processed()

    r.p("STRUKTURNA ANALIZA PRE PODELE PODATAKA")
    r.p("Faza 5a: gde se klase nalaze u vremenu (bez analize feature-target)")

    # ---------------------------------------------------------------- 1
    r.naslov("1. EPIZODE ZAUZETOSTI")
    ep = nadji_epizode(df)
    r.p(f"Ukupno redova   : {len(df)}")
    r.p(f"Ukupno epizoda  : {len(ep)}")
    r.p("")
    r.p("Po klasama:")
    po_klasi = ep.groupby("klasa").agg(
        epizoda=("redova", "size"),
        redova=("redova", "sum"),
        prosek_trajanja_min=("trajanje_min", "mean"),
        max_trajanje_min=("trajanje_min", "max"),
    )
    po_klasi["prosek_trajanja_min"] = po_klasi["prosek_trajanja_min"].round(1)
    r.p(po_klasi.to_string())

    r.p("")
    r.p("TUMACENJE:")
    r.p("  Identifikovane su 33 epizode zauzetosti. Unutar svake epizode postoji")
    r.p("  jaka vremenska zavisnost merenja - susedni redovi su gotovo identicni.")
    r.p("  Zato broj epizoda, a ne broj redova, odredjuje koliko razlicitih")
    r.p("  situacija je model zaista video za datu klasu.")

    # ---------------------------------------------------------------- 2
    r.naslov("2. EPIZODE KLASE 1 - najkriticnija klasa")
    ep1 = ep[ep["klasa"] == 1]
    r.p(f"Broj epizoda klase 1: {len(ep1)}")
    r.p("")
    r.p(ep1[["datum", "sesija", "pocetak", "kraj", "redova", "trajanje_min"]].to_string())

    # ---------------------------------------------------------------- 3
    r.naslov("3. IZBOR DUZINE BLOKA")
    r.p("Koliko blokova sadrzi bar jedan uzorak date klase, za razne duzine:")
    r.p("")
    tabela = {}
    for m in [10, 15, 20, 30, 45, 60]:
        tabela[f"{m} min"] = blokovi_po_velicini(df, m)
    t = pd.DataFrame(tabela)
    t.index.name = "klasa"
    r.p(t.to_string())

    ukupno_blokova = {}
    for m in [10, 15, 20, 30, 45, 60]:
        d = df.sort_values("Timestamp").copy()
        pocetak = d.groupby("sesija_id")["Timestamp"].transform("min")
        protekli = (d["Timestamp"] - pocetak).dt.total_seconds() / 60
        blok = d["sesija_id"].astype(str) + "_" + (protekli // m).astype(int).astype(str)
        ukupno_blokova[f"{m} min"] = blok.nunique()
    r.p("\nUkupan broj blokova:")
    r.p(pd.Series(ukupno_blokova).to_string())

    r.p("")
    r.p("KRITERIJUM IZBORA:")
    r.p("  Zelimo sto vise blokova sa klasom 1 (da podela bude moguca), ali blok")
    r.p("  ne sme biti kraci od trajanja tipicne epizode, jer bi se tada ista")
    r.p("  epizoda rasekla na train i test - bas curenje koje izbegavamo.")

    # ---------------------------------------------------------------- 4
    r.naslov("4. RASPORED KLASA PO SESIJAMA")
    unakrsno = pd.crosstab(df["sesija_id"], df[TARGET])
    unakrsno["ukupno"] = unakrsno.sum(axis=1)
    r.p(unakrsno.to_string())

    r.p("\nBroj epizoda po sesiji i klasi:")
    r.p(pd.crosstab(ep["sesija"], ep["klasa"]).to_string())

    izlaz = RESULTS_DIR / "03_struktura_epizoda.txt"
    r.snimi(izlaz)
    print(f"\n\nIzvestaj snimljen u: {izlaz}")
    return ep


# ====================================================================
# GRUPE ZA PODELU
# ====================================================================
def napravi_grupe(
    df: pd.DataFrame,
    max_grupa_min: int = 60,
    seci_klase: tuple[int, ...] = (0, 1, 2, 3),
) -> pd.DataFrame:
    """
    Pravi grupe koje su nedeljiva jedinica podele.

    Polazi se od epizode. Problem: epizode klase 0 traju i po 24 sata, pa bi
    jedna takva epizoda sama odnela 2779 redova u jedan skup i potpuno
    iskrivila proporcije. Zato se predugacke epizode seku na delove od
    najvise max_grupa_min minuta.

    Parametar seci_klase odredjuje kojim klasama je dozvoljeno sečenje.
    Sečenje epizode retke klase je OPASNO: dva dela iste epizode klase 1
    mogu zavrsiti u treningu i testu, a to je najgori moguci oblik curenja
    bas za klasu kojoj najmanje verujemo. Zato se u praksi seče samo klasa 0,
    koja ima 8228 redova i moze da podnese gubitak.
    """
    d = df.sort_values("Timestamp").reset_index(drop=True).copy()

    promena_klase = d[TARGET] != d[TARGET].shift()
    promena_sesije = d["sesija_id"] != d["sesija_id"].shift()
    d["epizoda_id"] = (promena_klase | promena_sesije).cumsum()

    # Redni broj poddela unutar epizode
    pocetak_ep = d.groupby("epizoda_id")["Timestamp"].transform("min")
    protekli_min = (d["Timestamp"] - pocetak_ep).dt.total_seconds() / 60
    poddeo = (protekli_min // max_grupa_min).astype(int)
    # Klase kojima secenje nije dozvoljeno ostaju jedna cela grupa
    poddeo = poddeo.where(d[TARGET].isin(seci_klase), 0)

    d["grupa_id"] = d["epizoda_id"].astype(str) + "_" + poddeo.astype(str)
    return d


# ====================================================================
# USVOJENA PODELA - test = cela sesija 2
# ====================================================================
def napravi_finalnu_podelu(purge_minuta: int = PURGE_MINUTA) -> pd.DataFrame:
    """
    Deli podatke na 'test', 'development' i 'purge'.

    Pravilo:
      test        - svi redovi iz sesija navedenih u TEST_SESIJE
      purge       - redovi iz ostalih sesija koji su vremenski unutar
                    purge_minuta od bilo kog test merenja
      development - sve ostalo

    Posto je test cela sesija snimanja, nijedna epizoda ne moze biti
    razdvojena izmedju development i test skupa.
    """
    df = load_processed()
    d = napravi_grupe(df)  # dodaje epizoda_id

    d["skup"] = "development"
    je_test = d["sesija_id"].isin(TEST_SESIJE)
    d.loc[je_test, "skup"] = "test"

    # Purge se racuna ZA SVAKU test sesiju posebno. Globalni min/max ne bi
    # valjao: sa dve test sesije promasio bi unutrasnje granice (kraj prve i
    # pocetak druge sesije).
    prozor = pd.Timedelta(minutes=purge_minuta)
    blizu = pd.Series(False, index=d.index)

    for sesija in TEST_SESIJE:
        ts = d.loc[d["sesija_id"] == sesija, "Timestamp"]
        if len(ts) == 0:
            continue
        blizu |= (d["Timestamp"] >= ts.min() - prozor) & (d["Timestamp"] <= ts.min())
        blizu |= (d["Timestamp"] >= ts.max()) & (d["Timestamp"] <= ts.max() + prozor)

    d.loc[blizu & ~je_test, "skup"] = "purge"
    return d


def sacuvaj_podelu(d: pd.DataFrame) -> None:
    """Snima tacne redove svakog skupa, da podela bude reproducibilna."""
    kolone = ["Timestamp", "datum", "sesija_id", "epizoda_id", TARGET, "skup"]
    izlaz = d[kolone].copy()
    izlaz.insert(0, "red_id", d.index)  # redni broj u hronoloski sortiranom skupu
    izlaz.to_csv(SPLIT_CSV, index=False)


def ucitaj_skup(naziv: str = "development", dozvoli_test: bool = False) -> pd.DataFrame:
    """
    Ucitava jedan skup iz zakljucane podele.

    Test skup je namerno zasticen: ucitava se samo ako se eksplicitno prosledi
    dozvoli_test=True. Time se sprecava da neki kasniji modul slucajno pogleda
    test podatke pri EDA, izboru atributa, izboru modela ili tuningu.
    """
    if naziv == "test" and not dozvoli_test:
        raise PermissionError(
            "Test skup (sesije 2 i 6) je zakljucan. Sme se ucitati samo u fazi "
            "konacne evaluacije, i to eksplicitno sa dozvoli_test=True."
        )
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Nema podele na {SPLIT_CSV}. Pokreni prvo: python pipeline.py --korak 4"
        )

    df = load_processed().sort_values("Timestamp").reset_index(drop=True)
    podela = pd.read_csv(SPLIT_CSV, parse_dates=["Timestamp"])
    podela = podela.sort_values("Timestamp").reset_index(drop=True)

    if not df["Timestamp"].equals(podela["Timestamp"]):
        raise ValueError(
            "Obradjeni skup i podela se ne poklapaju po vremenu. "
            "Pokreni ponovo: python pipeline.py"
        )

    df["skup"] = podela["skup"].values
    df["epizoda_id"] = podela["epizoda_id"].values
    return df[df["skup"] == naziv].drop(columns=["skup"]).reset_index(drop=True)


def _tabela_po_skupu(d: pd.DataFrame) -> pd.DataFrame:
    """Redovi i epizode po klasi, za svaki skup."""
    redovi = []
    for skup in ["development", "test", "purge"]:
        pod = d[d["skup"] == skup]
        if len(pod) == 0:
            continue
        red = {"skup": skup, "redova": len(pod),
               "%_dataseta": round(len(pod) / len(d) * 100, 1)}
        for k in [0, 1, 2, 3]:
            red[f"kl{k}_redova"] = int((pod[TARGET] == k).sum())
        for k in [0, 1, 2, 3]:
            red[f"kl{k}_epiz"] = int(pod.loc[pod[TARGET] == k, "epizoda_id"].nunique())
        redovi.append(red)
    return pd.DataFrame(redovi).set_index("skup")


# ====================================================================
# Provera 3-fold StratifiedGroupKFold na development skupu
# ====================================================================
def proveri_cv_foldove(d: pd.DataFrame, r: Izvestaj) -> bool:
    """
    Testira StratifiedGroupKFold sa sesijom kao grupom.

    Vraca True ako SVAKI fold ima smislenu evaluaciju svih klasa.
    Kriterijum: klasa u validaciji mora imati bar jednu celu epizodu
    I bar MIN_REDOVA_ZA_EVALUACIJU redova.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    dev = d[d["skup"] == "development"].copy()
    y = dev[TARGET].to_numpy()
    grupe = dev["sesija_id"].to_numpy()

    r.p(f"Development skup: {len(dev)} redova")
    r.p(f"Sesije u developmentu: {sorted(dev['sesija_id'].unique().tolist())}")
    r.p(f"Broj grupa (sesija)  : {dev['sesija_id'].nunique()}")
    r.p(f"Broj foldova         : {CV_FOLDOVA}")
    r.p("")
    r.p("Klasa 1 po sesijama u developmentu:")
    kl1 = dev[dev[TARGET] == 1].groupby("sesija_id").agg(
        redova=(TARGET, "size"),
        epizoda=("epizoda_id", "nunique"),
    )
    r.p(kl1.to_string())

    cv = StratifiedGroupKFold(n_splits=CV_FOLDOVA, shuffle=False)
    svi_ok = True

    for i, (idx_tr, idx_va) in enumerate(cv.split(dev, y, grupe), start=1):
        va = dev.iloc[idx_va]
        tr = dev.iloc[idx_tr]

        r.p("")
        r.p("-" * 74)
        r.p(f"FOLD {i}")
        r.p("-" * 74)
        r.p(f"  Sesije u VALIDACIJI: {sorted(va['sesija_id'].unique().tolist())}")
        r.p(f"  Sesije u TRENINGU  : {sorted(tr['sesija_id'].unique().tolist())}")
        r.p(f"  Redova validacija / trening: {len(va)} / {len(tr)}")
        r.p("")

        tab = []
        for k in [0, 1, 2, 3]:
            n_red = int((va[TARGET] == k).sum())
            n_ep = int(va.loc[va[TARGET] == k, "epizoda_id"].nunique())
            n_red_tr = int((tr[TARGET] == k).sum())
            upotrebljivo = (n_ep >= 1) and (n_red >= MIN_REDOVA_ZA_EVALUACIJU)
            if not upotrebljivo:
                svi_ok = False
            tab.append({
                "klasa": k,
                "val_redova": n_red,
                "val_epizoda": n_ep,
                "train_redova": n_red_tr,
                "upotrebljivo": "DA" if upotrebljivo else "NE",
            })
        r.p(pd.DataFrame(tab).set_index("klasa").to_string())

    return svi_ok


def finalna_podela() -> pd.DataFrame:
    """Faza 5 - pravi, snima i proverava usvojenu podelu."""
    ensure_dirs()
    r = Izvestaj()

    r.p("USVOJENA PODELA PODATAKA")
    r.p(f"Faza 5: test = cele sesije {list(TEST_SESIJE)}, development = ostale sesije")

    d = napravi_finalnu_podelu()

    # ---------------------------------------------------------------- 1
    r.naslov("1. PRAVILO PODELE")
    r.p(f"  Test sesije      : {list(TEST_SESIJE)}")
    r.p(f"  Purge margina    : {PURGE_MINUTA} min oko test sesije")
    r.p("")
    test_ts = d.loc[d["skup"] == "test", "Timestamp"]
    r.p(f"  Test vremenski opseg: {test_ts.min()}  ->  {test_ts.max()}")
    r.p("")
    r.p("  Obrazlozenje: obe test sesije ostaju CELE, pa nijedna epizoda nije")
    r.p("  razdvojena. Sesija 2 jedina sama sadrzi sve cetiri klase; sesija 6")
    r.p("  je jedna duga epizoda prazne sobe (24h) pa test dobija realan udeo")
    r.p("  klase 0. Sesija 5 ostaje u developmentu jer nosi veliku epizodu")
    r.p("  klase 3 (130 redova) pored one od 199 redova iz sesije 7.")

    # ---------------------------------------------------------------- 2
    r.naslov("2. KONACNA RASPODELA")
    r.p(_tabela_po_skupu(d).to_string())

    r.p("")
    r.p("Sesije po skupovima:")
    for skup in ["development", "test", "purge"]:
        ses = sorted(d.loc[d["skup"] == skup, "sesija_id"].unique().tolist())
        r.p(f"  {skup:12s}: {ses}")

    # Kljucna garancija
    po_ep = d.groupby("epizoda_id")["skup"].apply(
        lambda s: set(s.unique()) - {"purge"}
    )
    razdvojene = [int(e) for e, s in po_ep.items() if len(s) > 1]
    r.p("")
    r.p(f"Epizoda razdvojenih izmedju development i test: {len(razdvojene)} {razdvojene}")

    r.p("")
    r.p("Purge - koji redovi su izbaceni i zasto:")
    purge = d[d["skup"] == "purge"]
    if len(purge) == 0:
        r.p("  Nijedan red nije bilo potrebno izbaciti.")
    else:
        r.p(f"  {len(purge)} redova iz sesija {sorted(purge['sesija_id'].unique().tolist())}")
        r.p(f"  Vremenski opseg: {purge['Timestamp'].min()} -> {purge['Timestamp'].max()}")
        r.p(f"  Klase: {purge[TARGET].value_counts().sort_index().to_dict()}")

    # ---------------------------------------------------------------- 3
    r.naslov(f"3. PROVERA {CV_FOLDOVA}-FOLD StratifiedGroupKFold (grupa = sesija)")
    svi_ok = proveri_cv_foldove(d, r)

    r.p("")
    r.p("=" * 74)
    if svi_ok:
        r.p("ZAKLJUCAK: svaki fold smisleno evaluira sve cetiri klase.")
        r.p("Moze se nastaviti na EDA i modeliranje.")
    else:
        r.p("ZAKLJUCAK: NAJMANJE JEDAN FOLD NEMA UPOTREBLJIVU KLASU.")
        r.p("Ne nastavlja se na EDA/modeliranje dok se ne usvoji alternativa.")
    r.p("=" * 74)

    sacuvaj_podelu(d)
    r.p(f"\nPodela snimljena: {rel(SPLIT_CSV)}")

    izlaz = RESULTS_DIR / "04_finalna_podela.txt"
    r.snimi(izlaz)
    print(f"\n\nIzvestaj snimljen u: {izlaz}")
    return d


# ====================================================================
# CV FOLDOVI SA EPIZODOM KAO GRUPOM
# ====================================================================
def dodeli_epizode_foldovima(dev: pd.DataFrame, n_foldova: int = CV_FOLDOVA) -> pd.Series:
    """
    Rasporedjuje CELE epizode u foldove, stratifikovano po klasi.

    Za svaku klasu epizode se sortiraju po velicini opadajuce i svaka ide u
    fold koji je za tu klasu trenutno najmanje popunjen. Time najveće epizode
    retkih klasa zavrsavaju u RAZLICITIM foldovima, umesto da se nagomilaju
    u jednom - sto je kljucno za klasu 1.

    Deterministicki je: nema slucajnosti, isti ulaz daje isti raspored.
    """
    ep = dev.groupby("epizoda_id").agg(klasa=(TARGET, "first"), redova=(TARGET, "size"))

    dodela: dict[int, int] = {}
    for klasa, deo in ep.groupby("klasa"):
        kvota = deo["redova"].sum() / n_foldova
        trenutno = {f: 0 for f in range(n_foldova)}
        for ep_id, red in deo.sort_values("redova", ascending=False).iterrows():
            fold = min(trenutno, key=lambda f: trenutno[f] / kvota if kvota > 0 else 0)
            dodela[ep_id] = fold
            trenutno[fold] += red["redova"]

    return dev["epizoda_id"].map(dodela)


def purge_izmedju_foldova(dev: pd.DataFrame, fold_oznaka: pd.Series,
                          fold: int, purge_minuta: int = PURGE_MINUTA) -> pd.Series:
    """
    Vraca masku redova treninga koje treba izbaciti za dati fold.

    Izbacuje se svaki trening red koji je vremenski blizi od purge_minuta
    bilo kom validacionom redu tog folda. Tako se uklanja korelacija preko
    granice izmedju dve epizode koje su vremenski susedne.
    """
    je_val = fold_oznaka == fold
    val_ts = dev.loc[je_val, "Timestamp"].sort_values()
    tr_ts = dev.loc[~je_val, "Timestamp"]

    prozor = pd.Timedelta(minutes=purge_minuta)
    # Za svaki trening red najblizi validacioni trenutak
    poz = np.searchsorted(val_ts.values, tr_ts.values)
    razmak = pd.Series(pd.Timedelta.max, index=tr_ts.index)
    for pomeraj in (0, -1):
        idx = np.clip(poz + pomeraj, 0, len(val_ts) - 1)
        kandidat = pd.Series(np.abs(tr_ts.values - val_ts.values[idx]), index=tr_ts.index)
        razmak = pd.Series(np.minimum(razmak.values, kandidat.values), index=tr_ts.index)

    maska = pd.Series(False, index=dev.index)
    maska.loc[tr_ts.index] = razmak < prozor
    return maska


def predlozi_cv_foldove() -> bool:
    """
    Faza 5d - detaljan opis USVOJENOG CV-a na development skupu.

    Koristi se ista podela na foldove kao u treningu, tuningu i izboru
    atributa: StratifiedGroupKFold sa sesija_id kao grupom. Ovaj izvestaj
    dodaje detalj koji izvestaj o podeli nema - koje epizode padaju u koji
    fold i da li ijedan fold zavisi od mikro-epizode klase 1.

    Vraca True ako svaki fold smisleno evaluira sve klase.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    ensure_dirs()
    r = Izvestaj()

    d = napravi_finalnu_podelu()
    dev = d[d["skup"] == "development"].copy().reset_index(drop=True)

    r.p(f"CV FOLDOVI - grupa je SESIJA (sesija_id), {CV_FOLDOVA} folda")
    r.p(f"Faza 5d: test (sesije {list(TEST_SESIJE)}) je zakljucan i ne ucestvuje ni u cemu")
    r.p("")
    r.p("Ovo je ISTA podela na foldove koju koriste train.py, podesavanje")
    r.p("hiperparametara i izbor atributa - StratifiedGroupKFold(sesija_id).")

    # ---------------------------------------------------------------- 1
    r.naslov("1. DEVELOPMENT SKUP")
    r.p(f"Redova : {len(dev)}")
    r.p(f"Sesije : {sorted(dev['sesija_id'].unique().tolist())}")
    r.p(f"Epizoda: {dev['epizoda_id'].nunique()}")
    r.p("")
    r.p("Epizode u developmentu:")
    ep = dev.groupby("epizoda_id").agg(
        klasa=(TARGET, "first"),
        sesija=("sesija_id", "first"),
        redova=(TARGET, "size"),
        pocetak=("Timestamp", "min"),
    )
    ep["pocetak"] = ep["pocetak"].dt.strftime("%m-%d %H:%M")
    r.p(ep.to_string())

    r.p("\nEpizode klase 1 (najkriticnija klasa):")
    r.p(ep[ep["klasa"] == 1].to_string())

    # ---------------------------------------------------------------- 2
    r.naslov("2. RASPORED FOLDOVA")
    r.p("Sesije su razdvojene stvarnim prekidima snimanja, pa cela sesija ide")
    r.p("u jedan deo folda. Dodatna purge margina unutar foldova nije potrebna.")

    cv = StratifiedGroupKFold(n_splits=CV_FOLDOVA, shuffle=False)
    fold_oznaka = pd.Series(-1, index=dev.index, dtype=int)
    for f, (_, idx_va) in enumerate(
        cv.split(dev, dev[TARGET].to_numpy(), dev["sesija_id"].to_numpy())
    ):
        fold_oznaka.iloc[idx_va] = f
    dev = dev.assign(fold=fold_oznaka)

    sve_ok = True
    mikro_problem = []

    for f in range(CV_FOLDOVA):
        va = dev[dev["fold"] == f]
        tr = dev[dev["fold"] != f]

        r.p("")
        r.p("-" * 74)
        r.p(f"FOLD {f + 1}")
        r.p("-" * 74)
        r.p(f"  Validacija: {len(va)} redova | Trening: {len(tr)} redova")
        r.p(f"  Sesije u validaciji : {sorted(va['sesija_id'].unique().tolist())}")
        r.p(f"  Sesije u treningu   : {sorted(tr['sesija_id'].unique().tolist())}")
        r.p(f"  Epizode u validaciji: {sorted(va['epizoda_id'].unique().tolist())}")
        r.p("")

        redovi = []
        for k in [0, 1, 2, 3]:
            va_k = va[va[TARGET] == k]
            n_red, n_ep = len(va_k), va_k["epizoda_id"].nunique()
            upotrebljivo = (n_ep >= 1) and (n_red >= MIN_REDOVA_ZA_EVALUACIJU)
            if not upotrebljivo:
                sve_ok = False
            tr_k = tr[tr[TARGET] == k]
            redovi.append({
                "klasa": k,
                "val_redova": n_red,
                "val_epiz": n_ep,
                "val_epizode": sorted(va_k["epizoda_id"].unique().tolist()),
                "train_redova": len(tr_k),
                "train_epiz": tr_k["epizoda_id"].nunique(),
                "OK": "DA" if upotrebljivo else "NE",
            })
        r.p(pd.DataFrame(redovi).set_index("klasa").to_string())

        # Provera koju je trazila specifikacija: mikro-epizoda klase 1
        va1 = va[va[TARGET] == 1]
        if len(va1) > 0 and len(va1) < MIN_REDOVA_ZA_EVALUACIJU:
            mikro_problem.append(f + 1)

    # ---------------------------------------------------------------- 3
    r.naslov("3. PROVERA ZAVISNOSTI OD MIKRO-EPIZODE KLASE 1")
    r.p("Epizoda 4 (klasa 1, 22.12. 11:54:01-11:54:32) ima samo 2 reda.")
    r.p("Fold koji bi klasu 1 dobio SAMO preko nje ne bi je smisleno evaluirao.")
    r.p("")
    for f in range(CV_FOLDOVA):
        va1 = dev[(dev["fold"] == f) & (dev[TARGET] == 1)]
        eps = sorted(va1["epizoda_id"].unique().tolist())
        samo_mikro = eps == [4]
        r.p(f"  Fold {f + 1}: epizode klase 1 = {eps}, redova = {len(va1)}"
            f"{'   <-- SAMO MIKRO-EPIZODA' if samo_mikro else ''}")
    if mikro_problem:
        r.p(f"\n  Foldovi sa nedovoljno klase 1: {mikro_problem}")
    else:
        r.p("\n  Nijedan fold ne zavisi samo od mikro-epizode.")

    # ---------------------------------------------------------------- 4
    r.naslov("4. ZAKLJUCAK")
    if sve_ok:
        r.p("Svaki fold smisleno evaluira sve cetiri klase.")
    else:
        r.p("NEKI FOLD NEMA UPOTREBLJIVU KLASU - vidi oznake 'NE' iznad.")
        r.p("Ne prelazi se na EDA/modeliranje dok se ne usvoji resenje.")

    izlaz = RESULTS_DIR / "05_cv_foldovi.txt"
    r.snimi(izlaz)
    print(f"\n\nIzvestaj snimljen u: {izlaz}")
    return sve_ok


if __name__ == "__main__":
    finalna_podela()
    predlozi_cv_foldove()

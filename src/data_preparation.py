"""
FAZA 2 - Ucitavanje i pregled sirovog dataseta.

Ovaj modul u ovoj fazi SAMO CITA i OPISUJE podatke.
Nista se ne brise, ne popunjava i ne transformise - to dolazi u fazi 3
(preprocesiranje), tek posto vidimo sta podaci zaista sadrze.

Pokretanje:
    python src/data_preparation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RAW_CSV,
    PROCESSED_CSV,
    RESULTS_DIR,
    TARGET,
    DATE_COL,
    TIME_COL,
    SENSOR_COLS,
    SENSOR_GROUPS,
    EXPECTED_RANGES,
    SAMPLING_SECONDS,
    TIME_FEATURES,
    META_COLS,
    GAP_SECONDS,
    BLOCK_MINUTES,
    ensure_dirs,
    rel,
)

# Windows konzola podrazumevano nije UTF-8, pa nasa slova mogu da izazovu gresku
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


# ====================================================================
# Pomocna klasa: sve sto ispisemo istovremeno pamtimo za izvestaj u fajl
# ====================================================================
class Izvestaj:
    def __init__(self) -> None:
        self.linije: list[str] = []

    def p(self, tekst: str = "") -> None:
        print(tekst)
        self.linije.append(str(tekst))

    def naslov(self, tekst: str) -> None:
        self.p("")
        self.p("=" * 78)
        self.p(tekst)
        self.p("=" * 78)

    def snimi(self, putanja: Path) -> None:
        putanja.write_text("\n".join(self.linije), encoding="utf-8")


# ====================================================================
# Ucitavanje
# ====================================================================
def load_raw(putanja: Path = RAW_CSV) -> pd.DataFrame:
    """
    Ucitava sirovi CSV bez ikakvih izmena.

    Namerno NE parsiramo datum ovde - prvo hocemo da vidimo kako Date i Time
    zaista izgledaju kao tekst, pa tek onda odlucujemo kako ih obraditi.
    """
    if not putanja.exists():
        raise FileNotFoundError(
            f"Nije pronadjen CSV na putanji: {putanja}\n"
            f"Ocekuje se da se dataset nalazi u data/raw/."
        )
    df = pd.read_csv(putanja)
    # Zastita od skrivenih razmaka u zaglavlju CSV-a
    df.columns = [c.strip() for c in df.columns]
    return df


def dodaj_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spaja Date i Time u jednu datetime kolonu.

    Ovo NIJE feature za model - koristi se samo za analizu vremenske
    strukture (redosled merenja, razmaci, podela na dane).
    """
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(
        df[DATE_COL].astype(str).str.strip() + " " + df[TIME_COL].astype(str).str.strip(),
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce",
    )
    return df


# ====================================================================
# Sekcije pregleda
# ====================================================================
def _osnovne_informacije(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("1. OSNOVNE INFORMACIJE")
    r.p(f"Putanja fajla : {rel(RAW_CSV)}")
    r.p(f"Broj redova   : {df.shape[0]}")
    r.p(f"Broj kolona   : {df.shape[1]}")
    r.p(f"Zauzece u RAM : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    r.p("\nNazivi kolona:")
    for i, c in enumerate(df.columns, 1):
        r.p(f"  {i:2d}. {c}")

    r.p("\nOcekivane kolone iz specifikacije - provera:")
    ocekivane = [DATE_COL, TIME_COL] + SENSOR_COLS + [TARGET]
    nedostaju = [c for c in ocekivane if c not in df.columns]
    viska = [c for c in df.columns if c not in ocekivane]
    r.p(f"  Nedostaju : {nedostaju if nedostaju else 'nema'}")
    r.p(f"  Visak     : {viska if viska else 'nema'}")


def _tipovi_i_jedinstvene(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("2. TIPOVI PODATAKA I BROJ JEDINSTVENIH VREDNOSTI")
    info = pd.DataFrame(
        {
            "tip": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "nunique": df.nunique(),
            "primer": [df[c].iloc[0] for c in df.columns],
        }
    )
    r.p(info.to_string())


def _prvi_redovi(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("3. PRVIH 5 I POSLEDNJIH 5 REDOVA")
    r.p(df.head().to_string())
    r.p("...")
    r.p(df.tail().to_string())


def _nedostajuce(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("4. NEDOSTAJUCE VREDNOSTI")
    nan_po_koloni = df.isna().sum()
    ukupno = int(nan_po_koloni.sum())
    if ukupno == 0:
        r.p("Nema nedostajucih vrednosti ni u jednoj koloni.")
        r.p("=> Imputacija nije potrebna. Ne uvodimo vestacke NaN vrednosti")
        r.p("   samo da bi 'postojao' preprocessing.")
    else:
        r.p(nan_po_koloni[nan_po_koloni > 0].to_string())
        r.p(f"\nUkupno NaN celija: {ukupno}")

    # Prazni stringovi umeju da prodju kao validna vrednost, pa ih trazimo posebno.
    # Tekstualne kolone biramo po iskljucivanju da bi radilo i u pandas 2 i u 3.
    tekstualne = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if len(tekstualne) > 0:
        prazni = {c: int((df[c].astype(str).str.strip() == "").sum()) for c in tekstualne}
        prazni = {k: v for k, v in prazni.items() if v > 0}
        r.p(f"\nPrazni stringovi u tekstualnim kolonama: {prazni if prazni else 'nema'}")


def _duplikati(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("5. DUPLIKATI")
    potpuni = int(df.duplicated().sum())
    bez_vremena = int(df.duplicated(subset=SENSOR_COLS + [TARGET]).sum())

    r.p(f"Potpuno identicni redovi (sve kolone)        : {potpuni}")
    r.p(f"Identicna merenja bez Date/Time              : {bez_vremena}")

    r.p("")
    r.p("TUMACENJE:")
    r.p("  Potpuni duplikat bi znacio dva reda sa istim vremenom -> greska u")
    r.p("  logovanju. Ponovljena senzorska ocitavanja BEZ vremena su, medjutim,")
    r.p("  potpuno normalna za vremensku seriju: dok se u sobi nista ne desava,")
    r.p("  senzori vracaju iste brojeve. Takvi redovi NISU duplikati i ne")
    r.p("  smeju se brisati - nose informaciju o trajanju stanja.")


def _deskriptivna(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("6. DESKRIPTIVNA STATISTIKA NUMERICKIH ATRIBUTA")
    opis = df[SENSOR_COLS].describe().T
    opis["nunique"] = df[SENSOR_COLS].nunique()
    r.p(opis.to_string(float_format=lambda x: f"{x:12.3f}"))

    r.p("\nRasponi po grupama senzora:")
    for naziv, kolone in SENSOR_GROUPS.items():
        mn = df[kolone].min().min()
        mx = df[kolone].max().max()
        r.p(f"  {naziv:12s}: {mn:10.3f}  ...  {mx:10.3f}")
    r.p("\n=> Razliciti opsezi (npr. CO2 u stotinama ppm, zvuk oko 0-4) znace")
    r.p("   da ce logistickoj regresiji trebati skaliranje. Stabla ga ne trazе.")


def _ciljna_promenljiva(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("7. RASPODELA CILJNE PROMENLJIVE")
    broj = df[TARGET].value_counts().sort_index()
    proc = df[TARGET].value_counts(normalize=True).sort_index() * 100
    tab = pd.DataFrame({"broj_uzoraka": broj, "procenat_%": proc.round(2)})
    tab.index.name = "broj_osoba"
    r.p(tab.to_string())

    r.p(f"\nJedinstvene vrednosti targeta: {sorted(df[TARGET].unique().tolist())}")
    r.p(f"Odnos najbrojnija / najredja klasa: {broj.max() / broj.min():.1f} : 1")

    najveca_proc = proc.max()
    r.p("")
    r.p("TUMACENJE:")
    r.p(f"  Najzastupljenija klasa cini {najveca_proc:.1f}% podataka. To znaci da")
    r.p(f"  model koji uvek predvidja tu klasu dobija ~{najveca_proc:.1f}% accuracy")
    r.p("  bez ikakvog znanja. Zato accuracy sam po sebi nije dovoljna metrika -")
    r.p("  glavni kriterijum ce biti macro F1 (jednaka tezina svim klasama).")


def _analiza_datuma(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("8. ANALIZA KOLONA DATE I TIME")
    r.p(f"Tip kolone Date: {df[DATE_COL].dtype}   primer: {df[DATE_COL].iloc[0]!r}")
    r.p(f"Tip kolone Time: {df[TIME_COL].dtype}   primer: {df[TIME_COL].iloc[0]!r}")
    r.p(f"Broj razlicitih datuma: {df[DATE_COL].nunique()}")
    r.p(f"Datumi: {sorted(df[DATE_COL].unique().tolist())}")

    neuspesno = int(df["Timestamp"].isna().sum())
    r.p(f"\nNeuspesno parsiranih vremena: {neuspesno}")

    r.p("")
    r.p("ZAKLJUCAK:")
    r.p("  Date i Time su tekstualne kolone. U sirovom obliku NE ulaze u model -")
    r.p("  string '2017/12/22' modelu ne znaci nista, a numericki kodiran datum")
    r.p("  bi bio identifikator dana eksperimenta, ne fizicka velicina.")
    r.p("  Da li iz njih izvesti vremenske atribute (npr. sat u danu) proveravamo")
    r.p("  eksperimentalno u kasnijoj fazi, poredjenjem modela sa i bez njih.")


def _vremenska_struktura(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("9. VREMENSKA STRUKTURA (osnova za odluku o podeli podataka)")

    sortiran = df["Timestamp"].is_monotonic_increasing
    r.p(f"Da li su redovi hronoloski sortirani: {sortiran}")
    r.p(f"Duplirani vremenski pecati          : {int(df['Timestamp'].duplicated().sum())}")
    r.p(f"Prvo merenje : {df['Timestamp'].min()}")
    r.p(f"Zadnje merenje: {df['Timestamp'].max()}")

    d = df.sort_values("Timestamp")
    razmak = d["Timestamp"].diff().dt.total_seconds()

    r.p("\nRazmak izmedju uzastopnih merenja (sekunde):")
    r.p(razmak.describe().to_string(float_format=lambda x: f"{x:14.1f}"))
    r.p(f"\nDeklarisani period uzorkovanja iz opisa dataseta: {SAMPLING_SECONDS} s")
    r.p(f"Medijana stvarnog razmaka                       : {razmak.median():.1f} s")

    prag = SAMPLING_SECONDS * 4
    prekidi = razmak[razmak > prag]
    r.p(f"\nBroj prekida snimanja (razmak > {prag} s): {len(prekidi)}")
    if len(prekidi) > 0:
        r.p("Najveci prekidi (kraj -> nastavak snimanja):")
        for idx in prekidi.sort_values(ascending=False).head(10).index:
            poz = d.index.get_loc(idx)
            pre = d["Timestamp"].iloc[poz - 1]
            posle = d["Timestamp"].iloc[poz]
            r.p(f"  {pre}  ->  {posle}   ({razmak.loc[idx] / 3600:8.2f} h)")

    r.p("")
    r.p("ZASTO JE OVO KLJUCNO:")
    r.p("  Merenja su ~30 s razmaka, pa su susedni redovi gotovo identicni.")
    r.p("  Nasumicna podela (shuffle=True) bi rasporedila skoro iste redove i u")
    r.p("  trening i u test, pa bi model 'prepoznavao' vec vidjene trenutke.")
    r.p("  Rezultat bi bio lazno visok (blizu 100%) i metodoloski neispravan.")


def _klase_kroz_vreme(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("10. RASPORED KLASA KROZ VREME (da li je hronoloska podela moguca)")

    d = df.sort_values("Timestamp")
    dan = d["Timestamp"].dt.date

    pregled = d.groupby(dan).agg(
        merenja=("Timestamp", "size"),
        od=("Timestamp", "min"),
        do=("Timestamp", "max"),
        max_osoba=(TARGET, "max"),
        udeo_prazno_pct=(TARGET, lambda s: round((s == 0).mean() * 100, 1)),
    )
    pregled["od"] = pregled["od"].dt.strftime("%H:%M:%S")
    pregled["do"] = pregled["do"].dt.strftime("%H:%M:%S")
    pregled.index.name = "datum"
    r.p("Pregled po danima:")
    r.p(pregled.to_string())

    r.p("\nBroj uzoraka po klasi za svaki dan:")
    unakrsno = pd.crosstab(dan, d[TARGET])
    unakrsno.index.name = "datum"
    r.p(unakrsno.to_string())

    # Kljucna provera: da li bi poslednjih 20% podataka sadrzalo sve klase
    n = len(d)
    granica = int(n * 0.8)
    test_rep = d[TARGET].iloc[granica:]
    prisutne = sorted(test_rep.unique().tolist())
    sve_klase = sorted(df[TARGET].unique().tolist())

    r.p("")
    r.p("PROBNA PROVERA - da uzmemo poslednjih 20% redova kao test skup:")
    r.p(f"  Klase u tom delu : {prisutne}")
    r.p(f"  Sve klase u datasetu: {sve_klase}")
    if set(prisutne) == set(sve_klase):
        r.p("  => Sve klase su zastupljene. Cista hronoloska podela je izvodljiva.")
    else:
        nedostaju = sorted(set(sve_klase) - set(prisutne))
        r.p(f"  => PROBLEM: klase {nedostaju} ne postoje u tom test skupu.")
        r.p("     Naivna hronoloska podela bi bila neupotrebljiva za evaluaciju.")
        r.p("     U fazi 5 trazimo bolju strategiju (npr. podela po blokovima /")
        r.p("     sesijama, ili GroupKFold po danu) i obrazlazemo je.")


def _anomalije(r: Izvestaj, df: pd.DataFrame) -> None:
    r.naslov("11. PROVERA NELOGICNIH I EKSTREMNIH VREDNOSTI")

    r.p("A) Vrednosti van fizicki ocekivanog opsega:")
    nasli_problem = False
    for kol, (dmin, dmax) in EXPECTED_RANGES.items():
        maska = pd.Series(False, index=df.index)
        if dmin is not None:
            maska |= df[kol] < dmin
        if dmax is not None:
            maska |= df[kol] > dmax
        n = int(maska.sum())
        if n > 0:
            nasli_problem = True
            r.p(f"  {kol:15s} {n:5d} vrednosti van [{dmin}, {dmax}]"
                f"   min={df[kol].min():.3f} max={df[kol].max():.3f}")
    if not nasli_problem:
        r.p("  Nema vrednosti van ocekivanih fizickih opsega.")

    r.p("\nB) Konstantne kolone (nula informacije za model):")
    konstantne = [c for c in SENSOR_COLS if df[c].nunique() <= 1]
    r.p(f"  {konstantne if konstantne else 'nema'}")

    r.p("\nC) Binarni senzori - stvarne vrednosti:")
    for c in ["S6_PIR", "S7_PIR"]:
        vals = sorted(df[c].unique().tolist())
        udeo = (df[c] > 0).mean() * 100
        r.p(f"  {c}: vrednosti={vals}   udeo aktivnih={udeo:.1f}%")

    r.p("\nD) Broj ekstremnih vrednosti po IQR pravilu (informativno):")
    r.p("   VAZNO: ovo NIJE lista za brisanje. Kod senzora ekstremna vrednost")
    r.p("   najcesce znaci da se nesto stvarno desilo (upaljeno svetlo, ulazak")
    r.p("   osobe), a to je bas signal koji model treba da nauci.")
    for c in SENSOR_COLS:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            n_out = int(((df[c] < q1) | (df[c] > q3)).sum())
            oznaka = " (IQR=0)"
        else:
            n_out = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
            oznaka = ""
        r.p(f"  {c:15s} {n_out:6d}  ({n_out / len(df) * 100:5.2f} %){oznaka}")


# ====================================================================
# Glavna funkcija
# ====================================================================
def inspect_raw() -> pd.DataFrame:
    """Pokrece kompletan pregled sirovih podataka i snima izvestaj."""
    ensure_dirs()
    r = Izvestaj()

    r.p("PREGLED SIROVOG DATASETA - Room Occupancy Estimation")
    r.p("Faza 2 projekta: ucitavanje i provera podataka (bez izmena)")

    df = load_raw()
    df = dodaj_timestamp(df)

    _osnovne_informacije(r, df)
    _tipovi_i_jedinstvene(r, df)
    _prvi_redovi(r, df)
    _nedostajuce(r, df)
    _duplikati(r, df)
    _deskriptivna(r, df)
    _ciljna_promenljiva(r, df)
    _analiza_datuma(r, df)
    _vremenska_struktura(r, df)
    _klase_kroz_vreme(r, df)
    _anomalije(r, df)

    izlaz = RESULTS_DIR / "01_pregled_sirovih_podataka.txt"
    r.snimi(izlaz)
    print(f"\n\nIzvestaj snimljen u: {izlaz}")
    return df


# ====================================================================
# FAZA 3 - PREPROCESIRANJE
# ====================================================================
# Vodilja: menjamo samo ono za sta u fazi 2 postoji dokaz da treba menjati.
# Faza 2 je pokazala: nema NaN, nema potpunih duplikata, nema vrednosti van
# fizickih opsega. Dakle NE cistimo redove - nema sta da se cisti.
# Ostaje pravi posao: enkodiranje vremena i uklanjanje neupotrebljivih kolona.
# ====================================================================


def dodaj_vremenske_atribute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iz Timestamp-a izvodi numericke vremenske atribute.

    Zasto ne koristimo sirovi Date/Time:
      - '2017/12/22' je tekst; model iz njega ne moze nista da izracuna.
      - Kada bismo datum pretvorili u broj, to bi bio redni broj dana
        eksperimenta - identifikator, a ne fizicka velicina. Model bi mogao
        da nauci "10. januar => uglavnom prazno" umesto da cita senzore.

    Zasto sinus/kosinus:
      - Minut 1439 (23:59) i minut 0 (00:00) su vremenski susedni, ali kao
        obicni brojevi su maksimalno udaljeni. Ciklicno kodiranje to resava.
    """
    df = df.copy()
    minut = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute

    df["minut_u_danu"] = minut
    df["sat"] = df["Timestamp"].dt.hour
    df["dan_u_nedelji"] = df["Timestamp"].dt.dayofweek        # 0=ponedeljak
    df["vikend"] = (df["dan_u_nedelji"] >= 5).astype(int)
    df["minut_sin"] = np.sin(2 * np.pi * minut / 1440)
    df["minut_cos"] = np.cos(2 * np.pi * minut / 1440)
    return df


def dodaj_grupe_za_podelu(
    df: pd.DataFrame,
    gap_seconds: int = GAP_SECONDS,
    block_minutes: int = BLOCK_MINUTES,
) -> pd.DataFrame:
    """
    Dodaje kolone koje opisuju vremensku organizaciju podataka.

    Ove kolone NISU atributi za model - one su "adrese" redova koje ce faza 5
    koristiti da napravi podelu bez curenja podataka:

      datum     - kalendarski dan (za podelu po danima, strategija B)
      sesija_id - neprekidan blok snimanja; nova sesija pocinje kad je razmak
                  izmedju merenja veci od gap_seconds (nadjeno 7 takvih rupa)
      blok_id   - sesija isecena na intervale od block_minutes minuta
                  (za blok-podelu, strategija A)
    """
    df = df.sort_values("Timestamp").reset_index(drop=True).copy()

    razmak = df["Timestamp"].diff().dt.total_seconds()
    # cumsum preko logickog niza: svaki prekid povecava brojac sesije za 1
    df["sesija_id"] = (razmak > gap_seconds).fillna(False).cumsum().astype(int)

    df["datum"] = df["Timestamp"].dt.date.astype(str)

    # Redni broj bloka unutar sesije, racunat od pocetka te sesije
    pocetak_sesije = df.groupby("sesija_id")["Timestamp"].transform("min")
    protekli_min = (df["Timestamp"] - pocetak_sesije).dt.total_seconds() / 60
    redni_blok = (protekli_min // block_minutes).astype(int)
    df["blok_id"] = df["sesija_id"].astype(str) + "_" + redni_blok.astype(str)

    return df


def build_processed(sacuvaj: bool = True) -> pd.DataFrame:
    """
    Faza 3 - pravi obradjeni skup podataka i izvestaj o donetim odlukama.

    VAZNO: ovde se NE radi skaliranje. Skaler uci statistiku iz podataka
    (srednju vrednost i std), pa bi njegovo fitovanje na celom skupu prenelo
    informaciju iz testa u trening. Skaliranje ide u sklearn Pipeline u fazi 6,
    fitovano iskljucivo na trening delu.
    """
    ensure_dirs()
    r = Izvestaj()

    r.p("PREPROCESIRANJE - Room Occupancy Estimation")
    r.p("Faza 3 projekta: enkodiranje i priprema obradjenog skupa")

    df = load_raw()
    n_pre = len(df)
    df = dodaj_timestamp(df)

    # ---------------------------------------------------------------- 1
    r.naslov("1. ODLUKE O CISCENJU PODATAKA")
    r.p("Sve odluke slede iz nalaza faze 2 - nista se ne radi 'za svaki slucaj'.")
    r.p("")
    r.p(f"  Nedostajuce vrednosti  : {int(df.isna().sum().sum())}  -> imputacija NIJE potrebna")
    r.p(f"  Potpuni duplikati      : {int(df.duplicated().sum())}  -> nema sta da se ukloni")
    r.p("  Ponovljena ocitavanja  : 1301 (12.8%) -> ZADRZANI")
    r.p("      Razlog: to je prazna soba u kojoj se stanje ne menja. Brisanje bi")
    r.p("      uklonilo informaciju o tome koliko dugo je stanje trajalo i")
    r.p("      vestacki promenilo raspodelu klasa.")
    r.p("  Ekstremne vrednosti    : ZADRZANE")
    r.p("      Razlog: nijedna nije van fizicki mogucih opsega. Kod senzora skok")
    r.p("      u svetlosti ili zvuku znaci da se dogadjaj stvarno desio - to je")
    r.p("      bas signal koji model treba da nauci, a ne sum.")
    r.p("")
    r.p("  => Nijedan red se ne uklanja u ovoj fazi.")

    # ---------------------------------------------------------------- 2
    r.naslov("2. ENKODIRANJE")
    r.p("A) Ciljna promenljiva Room_Occupancy_Count")
    r.p(f"   Vec je ceo broj: {sorted(df[TARGET].unique().tolist())}")
    r.p("   => LabelEncoder nije potreban; klase su spremne za sklearn.")

    r.p("\nB) PIR senzori S6_PIR, S7_PIR")
    r.p("   Vec binarni 0/1. => One-hot enkodiranje nema smisla.")

    r.p("\nC) Kategorijske kolone")
    r.p("   Ne postoje - svi senzorski atributi su numericki.")

    r.p("\nD) Date i Time (tekstualne kolone) -> vremenski atributi")
    df = dodaj_vremenske_atribute(df)
    for f in TIME_FEATURES:
        r.p(f"   {f:15s} opseg [{df[f].min():.3f}, {df[f].max():.3f}]")
    r.p("   minut_sin i minut_cos: ciklicno kodiranje doba dana, da bi 23:59 i")
    r.p("   00:00 bili blizu jedan drugom, a ne na suprotnim krajevima skale.")

    # ---------------------------------------------------------------- 3
    r.naslov("3. UKLONJENI ATRIBUTI")
    r.p("  Date  - tekst; zamenjen izvedenim vremenskim atributima")
    r.p("  Time  - tekst; zamenjen izvedenim vremenskim atributima")
    r.p("")
    r.p("  Timestamp se ZADRZAVA u fajlu, ali kao meta-podatak za podelu i")
    r.p("  crtanje grafika - NE kao ulaz u model.")

    # ---------------------------------------------------------------- 4
    r.naslov("4. VREMENSKE GRUPE ZA PODELU BEZ CURENJA PODATAKA")
    df = dodaj_grupe_za_podelu(df)
    r.p(f"  Prag za novu sesiju : razmak > {GAP_SECONDS} s")
    r.p(f"  Duzina bloka        : {BLOCK_MINUTES} min")
    r.p(f"  Broj sesija snimanja: {df['sesija_id'].nunique()}")
    r.p(f"  Broj blokova        : {df['blok_id'].nunique()}")
    r.p(f"  Broj kalendarskih dana: {df['datum'].nunique()}")

    r.p("\nSesije snimanja:")
    ses = df.groupby("sesija_id").agg(
        od=("Timestamp", "min"),
        do=("Timestamp", "max"),
        redova=("Timestamp", "size"),
        klase=(TARGET, lambda s: sorted(s.unique().tolist())),
    )
    r.p(ses.to_string())

    r.p("\nKoliko blokova sadrzi bar jedan uzorak date klase:")
    for k in sorted(df[TARGET].unique()):
        n_blok = df.loc[df[TARGET] == k, "blok_id"].nunique()
        r.p(f"  klasa {k}: {n_blok:4d} blokova od ukupno {df['blok_id'].nunique()}")
    r.p("\n  Ovo je ulaz za fazu 5: pokazuje da li blok-podela moze da obezbedi")
    r.p("  sve cetiri klase u svakom od skupova train/val/test.")

    # ---------------------------------------------------------------- 5
    r.naslov("5. SKALIRANJE - ZASTO GA OVDE NEMA")
    r.p("  StandardScaler uci srednju vrednost i standardnu devijaciju IZ podataka.")
    r.p("  Kada bismo ga fitovali sada, na celom skupu, statistika test podataka")
    r.p("  bi usla u transformaciju trening podataka => curenje informacija.")
    r.p("")
    r.p("  Zato skaliranje ide u sklearn Pipeline (faza 6), fitovano samo na")
    r.p("  trening delu, i to samo za modele kojima treba (logisticka regresija).")
    r.p("  Stablima (Decision Tree, Random Forest, XGBoost) skaliranje ne treba,")
    r.p("  jer dele podatke po pragovima i monotone transformacije ih ne menjaju.")

    # ---------------------------------------------------------------- 6
    r.naslov("6. REZULTAT")
    izlazne = META_COLS + SENSOR_COLS + TIME_FEATURES + [TARGET]
    obradjen = df[izlazne].copy()

    r.p(f"  Redova pre obrade  : {n_pre}")
    r.p(f"  Redova posle obrade: {len(obradjen)}   (razlika: {len(obradjen) - n_pre})")
    r.p(f"  Kolona u izlazu    : {len(izlazne)}")
    r.p("")
    r.p(f"  Meta-kolone ({len(META_COLS)}): {META_COLS}")
    r.p(f"  Senzorski atributi ({len(SENSOR_COLS)}): {SENSOR_COLS}")
    r.p(f"  Vremenski atributi ({len(TIME_FEATURES)}): {TIME_FEATURES}")
    r.p(f"  Ciljna promenljiva: {TARGET}")
    r.p("")
    r.p(f"  Provera NaN u izlazu: {int(obradjen.isna().sum().sum())}")

    if sacuvaj:
        obradjen.to_csv(PROCESSED_CSV, index=False)
        r.p(f"\n  Snimljeno: {rel(PROCESSED_CSV)}")

    r.snimi(RESULTS_DIR / "02_preprocesiranje_izvestaj.txt")
    print(f"\n\nIzvestaj snimljen u: {RESULTS_DIR / '02_preprocesiranje_izvestaj.txt'}")
    return obradjen


def load_processed() -> pd.DataFrame:
    """Ucitava obradjeni skup; koriste ga faze 4 i dalje."""
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"Nema obradjenog skupa na {PROCESSED_CSV}. "
            f"Pokreni prvo: python pipeline.py --korak 2"
        )
    df = pd.read_csv(PROCESSED_CSV, parse_dates=["Timestamp"])
    return df


if __name__ == "__main__":
    inspect_raw()
    build_processed()

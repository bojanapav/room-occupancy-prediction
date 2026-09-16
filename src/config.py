"""
Centralno mesto za putanje i konstante celog projekta.

Razlog postojanja ovog fajla: nijedan drugi modul ne sme da sadrzi
"hardkodiranu" putanju. Ako se projekat premesti, menja se samo ovde.
"""

from pathlib import Path

# ---------------------------------------------------------------- putanje
# parents[1] = folder Projekat_Room_Occupancy (jer je ovaj fajl u src/)
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RAW_CSV = DATA_RAW_DIR / "Occupancy_Estimation.csv"


def rel(putanja: Path) -> str:
    """
    Vraca putanju relativnu u odnosu na koren projekta.

    Koristi se pri ispisu u izvestaje, da u njima ne zavrsi apsolutna
    lokalna putanja sa imenom korisnika - izvestaji idu u javni repozitorijum.
    """
    try:
        return str(Path(putanja).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(putanja)


def ensure_dirs() -> None:
    """Pravi izlazne foldere ako ne postoje (poziva se iz skripti)."""
    for d in (DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- kolone
TARGET = "Room_Occupancy_Count"

DATE_COL = "Date"
TIME_COL = "Time"

TEMP_COLS = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp"]
LIGHT_COLS = ["S1_Light", "S2_Light", "S3_Light", "S4_Light"]
SOUND_COLS = ["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]
CO2_COLS = ["S5_CO2", "S5_CO2_Slope"]
PIR_COLS = ["S6_PIR", "S7_PIR"]

# Svi sirovi senzorski atributi (bez vremena i bez targeta)
SENSOR_COLS = TEMP_COLS + LIGHT_COLS + SOUND_COLS + CO2_COLS + PIR_COLS

# Grupe senzora - koristi se u EDA za proveru redundanse unutar grupe
SENSOR_GROUPS = {
    "Temperatura": TEMP_COLS,
    "Svetlost": LIGHT_COLS,
    "Zvuk": SOUND_COLS,
    "CO2": CO2_COLS,
    "PIR": PIR_COLS,
}

# Fizicki ocekivani opsezi - koriste se SAMO za detekciju sumnjivih vrednosti,
# ne za automatsko brisanje. None znaci "granica nije definisana".
EXPECTED_RANGES = {
    **{c: (10.0, 40.0) for c in TEMP_COLS},      # temperatura u zatvorenom, C
    **{c: (0.0, None) for c in LIGHT_COLS},      # lux ne moze biti negativan
    **{c: (0.0, None) for c in SOUND_COLS},      # volt/amplituda, ne moze < 0
    "S5_CO2": (300.0, None),                     # spoljasnji vazduh ~400 ppm
    "S5_CO2_Slope": (None, None),                # nagib SME biti negativan
    **{c: (0.0, 1.0) for c in PIR_COLS},         # binarni senzor pokreta
}

# ---------------------------------------------------------------- ostalo
# Deklarisani period uzorkovanja iz opisa dataseta (proverava se u fazi 2)
SAMPLING_SECONDS = 30

RANDOM_STATE = 42

# ------------------------------------------- rezultat faze 7 (tuning)
# Najbolji hiperparametri nadjeni GridSearchCV-om na development skupu,
# uz usvojenu 2-fold podelu po sesija_id i kriterijum macro F1.
# Ovde stoje da bi kasnije faze koristile TACNO istu konfiguraciju.
DT_NAJBOLJI = {
    "class_weight": "balanced",
    "criterion": "entropy",
    "max_depth": 3,
    "min_samples_leaf": 5,
    "min_samples_split": 2,
}

# ------------------------------------- ZAKLJUCANA FINALNA KONFIGURACIJA
# Odluka doneta iskljucivo na development skupu, PRE gledanja testa.
# Posle finalne evaluacije se vise ne menja - ni model, ni atributi,
# ni hiperparametri, ni podela.
FINALNI_ATRIBUTI = ["S2_Light", "S1_Light", "S3_Light", "S4_Light", "S2_Temp"]

RF_NAJBOLJI = {
    "class_weight": "balanced",
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 10,
    "n_estimators": 100,
}

# ---------------------------------------------- obradjeni skup (faza 3)
PROCESSED_CSV = DATA_PROCESSED_DIR / "occupancy_processed.csv"

# Vremenski atributi izvedeni iz Date/Time.
# NAPOMENA: ne ulaze automatski u model. U fazi 6 treniramo dve varijante
# (sa i bez njih) i poredimo, jer postoji rizik da model nauci raspored
# eksperimenta umesto fizicke veze senzor -> broj ljudi.
TIME_FEATURES = [
    "minut_u_danu",
    "sat",
    "dan_u_nedelji",
    "vikend",
    "minut_sin",
    "minut_cos",
]

# Kolone koje sluze za organizaciju podataka i podelu train/test.
# NIKADA ne ulaze u model kao atributi - to obezbedjuje get_features().
META_COLS = ["Timestamp", "datum", "sesija_id", "blok_id"]

# Osnovni skup atributa = samo stvarna senzorska merenja.
# Vremenski atributi su KANDIDATI, ne usvojeni feature-i. U fazi 6 poredimo
# obe varijante da bismo proverili da li model uci fiziku ili raspored
# eksperimenta.
FEATURE_SETS = {
    "senzori": SENSOR_COLS,
    "senzori_plus_vreme": SENSOR_COLS + TIME_FEATURES,
}


def get_features(skup: str = "senzori") -> list[str]:
    """
    Vraca listu atributa za dati skup i garantuje da meta-kolone i target
    nikada ne mogu da udju u model.

    Namerno je napisano kao provera koja podize gresku: komentar se moze
    prevideti, provera ne moze.
    """
    if skup not in FEATURE_SETS:
        raise ValueError(f"Nepoznat skup atributa: {skup}. Dostupno: {list(FEATURE_SETS)}")

    feats = list(FEATURE_SETS[skup])

    zabranjeno = set(META_COLS) | {TARGET, DATE_COL, TIME_COL}
    presek = zabranjeno.intersection(feats)
    if presek:
        raise ValueError(
            f"Zabranjene kolone su se nasle medju atributima: {sorted(presek)}. "
            f"Meta-kolone i target ne smeju biti ulaz modela."
        )
    return feats

# Prag za detekciju prekida snimanja: razmak veci od ovoga znaci nova sesija
GAP_SECONDS = SAMPLING_SECONDS * 4  # 120 s

# Duzina bloka za blok-podelu podataka (faza 5)
BLOCK_MINUTES = 30

# ------------------------------------------------ USVOJENA PODELA (faza 5)
# Test skup = cele sesije snimanja 2 i 6.
# Obrazlozenje:
#   - obe sesije ostaju CELE, pa nijedna epizoda nije razdvojena
#   - sesija 2 jedina sama sadrzi sve cetiri klase
#   - sesija 6 je jedna duga epizoda klase 0 (2779 redova, 24h prazne sobe),
#     pa test dobija realan udeo prazne prostorije
#   - sesija 5 OSTAJE u developmentu, jer nosi veliku epizodu klase 3
#     (130 redova) pored one od 199 redova iz sesije 7
TEST_SESIJE = (2, 6)

# Purge margina oko test sesije. Merenja su na 30 s razmaka, pa su redovi tik
# uz granicu korelisani sa test podacima i izbacuju se iz developmenta.
PURGE_MINUTA = 5

# Broj foldova za unakrsnu validaciju na development skupu.
# 2 umesto 3: klasa 3 u developmentu ima samo dve velike epizode (130 i 199
# redova), pa se vise od dva folda ne moze popuniti smisleno.
CV_FOLDOVA = 2

# Fajl sa zakljucanom podelom
SPLIT_CSV = DATA_PROCESSED_DIR / "podela.csv"

# Prag upotrebljivosti klase u validacionom delu folda.
# Klasa se smatra smisleno evaluiranom ako u validaciji ima bar jednu celu
# epizodu I bar ovoliko redova (30 redova ~ 15 min snimanja).
MIN_REDOVA_ZA_EVALUACIJU = 30

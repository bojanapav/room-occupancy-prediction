"""
FAZA 10a - Finalno treniranje i eksport modela.

Model se trenira na CELOM development skupu (6608 redova), potpuno isto kao
onaj koji je evaluiran na testu u fazi 9. Namerno se NE trenira na test skupu
ni na svih 10129 redova - da bi eksportovani model bio tacno onaj na koji se
odnose prijavljeni finalni rezultati.

Uz model se snima sve sto je potrebno da predikcija bude reproducibilna:
  - tacan REDOSLED ulaznih atributa (kljucno: sklearn ocekuje isti redosled
    kolona kao pri treningu, a niz brojeva sam po sebi ne nosi imena)
  - klase koje model poznaje
  - hiperparametri
  - opsezi vrednosti iz trening podataka (da UI ne mora nista da pretpostavlja)

Pokretanje:
    python src/export_model.py
"""

from __future__ import annotations

import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from config import (
    MODELS_DIR,
    RESULTS_DIR,
    TARGET,
    RANDOM_STATE,
    DT_NAJBOLJI,
    FINALNI_ATRIBUTI,
    ensure_dirs,
    rel,
)
from data_preparation import Izvestaj
from data_split import ucitaj_skup

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_FAJL = MODELS_DIR / "finalni_model.joblib"


def eksportuj_model() -> dict:
    ensure_dirs()
    r = Izvestaj()

    r.p("EKSPORT KONACNOG MODELA")
    r.p("Faza 10a - deployment")

    # ---------------------------------------------------------------- 1
    dev = ucitaj_skup("development")
    X = dev[FINALNI_ATRIBUTI]
    y = dev[TARGET]

    r.naslov("1. TRENING")
    r.p("Model se trenira na CELOM development skupu - isti podaci i ista")
    r.p("konfiguracija kao model evaluiran na testu u fazi 9.")
    r.p("")
    r.p(f"  Redova    : {len(dev)}")
    r.p(f"  Sesije    : {sorted(dev['sesija_id'].unique().tolist())}")
    r.p(f"  Period    : {dev['Timestamp'].min()} -> {dev['Timestamp'].max()}")
    r.p(f"  Atributi  : {FINALNI_ATRIBUTI}")
    r.p("")
    r.p("  Test skup NIJE ukljucen u trening.")

    model = DecisionTreeClassifier(random_state=RANDOM_STATE, **DT_NAJBOLJI)
    model.fit(X, y)

    # ---------------------------------------------------------------- 2
    r.naslov("2. SADRZAJ EKSPORTOVANOG PAKETA")

    opsezi = {
        kolona: {
            "min": float(dev[kolona].min()),
            "max": float(dev[kolona].max()),
            "medijana": float(dev[kolona].median()),
            "decimala": 2 if dev[kolona].dtype.kind == "f" else 0,
        }
        for kolona in FINALNI_ATRIBUTI
    }

    paket = {
        "model": model,
        "atributi": list(FINALNI_ATRIBUTI),      # REDOSLED je obavezujuci
        "klase": [int(k) for k in model.classes_],
        "hiperparametri": dict(DT_NAJBOLJI),
        "opsezi": opsezi,
        "metapodaci": {
            "tip_modela": "DecisionTreeClassifier",
            "trenirano_na": "development skup (sesije 0, 1, 4, 5, 7)",
            "broj_redova_treninga": int(len(dev)),
            "skaliranje": "nije primenjeno - stablo odlucivanja ga ne zahteva",
            "datum_eksporta": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "napomena": (
                "Ulazi se modelu MORAJU proslediti u redosledu iz polja "
                "'atributi'. Model pamti imena kolona iz treninga "
                "(feature_names_in_) i proverava ih kada dobije DataFrame, ali "
                "ako dobije obican niz brojeva oslanja se samo na pozicije i "
                "tiho bi prihvatio pogresan redosled."
            ),
        },
    }

    for kljuc in ("atributi", "klase", "hiperparametri"):
        r.p(f"  {kljuc:16s}: {paket[kljuc]}")
    r.p(f"  {'opsezi':16s}: min/max/medijana za svaki atribut (koristi ih UI)")
    r.p(f"  {'metapodaci':16s}: {len(paket['metapodaci'])} polja")

    r.p("")
    r.p("Opsezi iz trening podataka:")
    r.p(pd.DataFrame(opsezi).T.to_string())

    joblib.dump(paket, MODEL_FAJL)
    r.p("")
    r.p(f"Snimljeno: {rel(MODEL_FAJL)}")
    r.p(f"Velicina : {MODEL_FAJL.stat().st_size / 1024:.1f} KB")

    # ---------------------------------------------------------------- 3
    r.naslov("3. PROVERA UCITAVANJA I PREDIKCIJE")
    ucitan = joblib.load(MODEL_FAJL)

    provere = []

    provere.append(("model se ucitava",
                    hasattr(ucitan["model"], "predict")))
    provere.append(("redosled atributa sacuvan",
                    ucitan["atributi"] == list(FINALNI_ATRIBUTI)))
    provere.append(("klase su 0-3",
                    ucitan["klase"] == [0, 1, 2, 3]))
    provere.append(("hiperparametri sacuvani",
                    ucitan["hiperparametri"] == dict(DT_NAJBOLJI)))

    # Predikcija ucitanog modela mora biti identicna predikciji originalnog
    pred_original = model.predict(X)
    pred_ucitan = ucitan["model"].predict(X[ucitan["atributi"]])
    provere.append(("ucitani model daje identicne predikcije",
                    bool(np.array_equal(pred_original, pred_ucitan))))

    # Predikcija iz jednog reda, tacno onako kako to radi Streamlit aplikacija:
    # DataFrame sa imenima kolona u sacuvanom redosledu.
    primer_red = X.iloc[0]
    ulaz = pd.DataFrame([[float(primer_red[a]) for a in ucitan["atributi"]]],
                        columns=ucitan["atributi"])
    pred_jedan = ucitan["model"].predict(ulaz)
    provere.append(("predikcija jednog reda (kao u aplikaciji) radi",
                    int(pred_jedan[0]) == int(pred_original[0])))

    # Kontrola 1: pogresan redosled kolona menja rezultat.
    # Prosledjuje se kao numpy niz, jer bi DataFrame sa pogresnim redosledom
    # sklearn odmah odbio - sto je samo po sebi dodatna zastita.
    obrnut = list(reversed(ucitan["atributi"]))
    pred_obrnut = ucitan["model"].predict(X[obrnut].to_numpy())
    provere.append(("pogresan redosled menja rezultat (dokaz da redosled znaci)",
                    not bool(np.array_equal(pred_original, pred_obrnut))))

    # Kontrola 2: DataFrame sa pogresnim redosledom mora da izazove gresku
    try:
        ucitan["model"].predict(X[obrnut])
        odbija_pogresan = False
    except Exception:
        odbija_pogresan = True
    provere.append(("model odbija DataFrame sa pogresnim redosledom kolona",
                    odbija_pogresan))

    for naziv, prosao in provere:
        r.p(f"  [{'OK' if prosao else 'GRESKA'}] {naziv}")

    sve_ok = all(p for _, p in provere)
    r.p("")
    r.p("SVE PROVERE PROSLE" if sve_ok else "NEKA PROVERA NIJE PROSLA")

    r.p("")
    r.p("Primer predikcije:")
    r.p(f"  ulaz  = {ulaz.iloc[0].to_dict()}")
    r.p(f"  izlaz = {int(pred_jedan[0])} osoba")

    r.naslov("4. KAKO SE MODEL KORISTI")
    r.p("  import joblib, pandas as pd")
    r.p("  paket = joblib.load('models/finalni_model.joblib')")
    r.p("  red = pd.DataFrame([vrednosti], columns=paket['atributi'])")
    r.p("  broj_osoba = paket['model'].predict(red)[0]")
    r.p("")
    r.p("  Prosledjivanje DataFrame-a sa imenima kolona je bezbednije od golog")
    r.p("  niza: sklearn tada sam proverava da li se redosled poklapa sa")
    r.p("  treningom i javlja gresku ako se ne poklapa.")

    r.snimi(RESULTS_DIR / "13_eksport_modela.txt")
    print(f"\n\nModel    : {MODEL_FAJL}")
    print(f"Izvestaj : {RESULTS_DIR / '13_eksport_modela.txt'}")

    if not sve_ok:
        raise RuntimeError("Provera eksportovanog modela nije prosla.")
    return paket


if __name__ == "__main__":
    eksportuj_model()

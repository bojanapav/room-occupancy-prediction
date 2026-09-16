"""
FAZA 6 - Odabir i pocetno treniranje modela.

Cilj ove faze: istrenirati osnovne verzije cetiri modela iz specifikacije i
videti kako se ponasaju. NEMA podesavanja hiperparametara i NEMA izbora
atributa - to su zasebne faze.

Postovanje zakljucane podele:
  - test (sesije 2 i 6) se NE ucitava; ucitavac ga odbija bez eksplicitne dozvole
  - radi se iskljucivo na development skupu (sesije 0, 1, 4, 5, 7)
  - poredjenje ide preko usvojenog 2-fold CV-a sa sesija_id kao grupom
  - StandardScaler je unutar sklearn Pipeline-a, pa se fituje samo na
    trening delu svakog folda

Pokretanje:
    python src/train.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    RESULTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    TARGET,
    RANDOM_STATE,
    CV_FOLDOVA,
    get_features,
    ensure_dirs,
)
from data_preparation import Izvestaj
from data_split import ucitaj_skup

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

KLASE = [0, 1, 2, 3]


# ====================================================================
# Modeli
# ====================================================================
def napravi_modele(balansirano: bool) -> dict[str, tuple[object, bool]]:
    """
    Vraca {naziv: (model, treba_sample_weight)}.

    Svi modeli su u osnovnoj postavci - menja se samo nacin tretiranja
    nebalansiranih klasa:
      balansirano=False -> podrazumevano ponasanje biblioteke
      balansirano=True  -> class_weight='balanced' (XGBoost nema taj parametar,
                           pa se kod njega koristi ekvivalentan sample_weight)

    VAZNO: ovo NIJE podesavanje hiperparametara. To su dve razumne pocetne
    postavke, jer pri 80% udela klase 0 podrazumevani model moze uopste da
    ne predvidi retke klase.
    """
    cw = "balanced" if balansirano else None

    return {
        # Logistickoj regresiji treba skaliranje - zato ide u Pipeline.
        # Skaler se fituje unutar fit() svakog folda, dakle samo na treningu.
        "LogisticRegression": (
            Pipeline([
                ("skaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    class_weight=cw,
                    random_state=RANDOM_STATE,
                )),
            ]),
            False,
        ),
        # Stablima skaliranje ne treba - dele podatke po pragovima.
        "DecisionTree": (
            DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_STATE),
            False,
        ),
        "RandomForest": (
            RandomForestClassifier(
                class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                objective="multi:softprob",
                num_class=len(KLASE),
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            balansirano,
        ),
    }


def napravi_baseline() -> dict[str, tuple[object, bool]]:
    """
    Baseline za poredjenje - NIJE jedan od glavnih modela.

    'most_frequent' uvek predvidja najcescu klasu (prazna soba). Pokazuje
    koliku tacnost model dobija bez ikakvog znanja o senzorima.
    """
    return {
        "Baseline (najcesca klasa)": (
            DummyClassifier(strategy="most_frequent"), False
        )
    }


# ====================================================================
# Unakrsna validacija
# ====================================================================
def _fituj(model, X, y, sample_weight=None):
    """Fituje model; za Pipeline prosledjuje tezine poslednjem koraku."""
    if sample_weight is None:
        model.fit(X, y)
    elif isinstance(model, Pipeline):
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": sample_weight})
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def foldovi(dev: pd.DataFrame):
    """Usvojeni 2-fold CV: StratifiedGroupKFold sa sesija_id kao grupom."""
    cv = StratifiedGroupKFold(n_splits=CV_FOLDOVA, shuffle=False)
    return list(cv.split(dev, dev[TARGET].to_numpy(), dev["sesija_id"].to_numpy()))


def oceni_model(dev: pd.DataFrame, atributi: list[str], model_fn, treba_tezine: bool):
    """
    Trenira model na svakom foldu i vraca:
      - listu macro F1 po foldu
      - listu balanced accuracy po foldu
      - listu accuracy po foldu
      - objedinjene out-of-fold predikcije (svaki red tacno jednom)
      - ukupno vreme treniranja
    """
    X = dev[atributi].to_numpy()
    y = dev[TARGET].to_numpy()

    oof = np.full(len(dev), -1, dtype=int)
    macro_f1, bal_acc, acc = [], [], []
    trajanje = 0.0

    for idx_tr, idx_va in foldovi(dev):
        model = model_fn()
        tezine = (
            compute_sample_weight("balanced", y[idx_tr]) if treba_tezine else None
        )
        pocetak = time.perf_counter()
        _fituj(model, X[idx_tr], y[idx_tr], tezine)
        trajanje += time.perf_counter() - pocetak

        pred = model.predict(X[idx_va])
        oof[idx_va] = pred

        macro_f1.append(f1_score(y[idx_va], pred, average="macro", zero_division=0))
        bal_acc.append(balanced_accuracy_score(y[idx_va], pred))
        acc.append(accuracy_score(y[idx_va], pred))

    return macro_f1, bal_acc, acc, oof, trajanje


# ====================================================================
# Izvestavanje
# ====================================================================
def _tabela_po_klasama(y_true, y_pred) -> pd.DataFrame:
    p, r, f, n = precision_recall_fscore_support(
        y_true, y_pred, labels=KLASE, zero_division=0
    )
    return pd.DataFrame(
        {"precision": p.round(3), "recall": r.round(3),
         "f1": f.round(3), "uzoraka": n},
        index=pd.Index(KLASE, name="klasa"),
    )


def _proveri_foldove(dev: pd.DataFrame, r: Izvestaj) -> None:
    """Ispisuje sastav foldova i potvrdjuje da su sesije cele u jednom delu."""
    r.p("Usvojeni CV: StratifiedGroupKFold, grupa = sesija_id, "
        f"{CV_FOLDOVA} folda")
    for i, (idx_tr, idx_va) in enumerate(foldovi(dev), 1):
        va, tr = dev.iloc[idx_va], dev.iloc[idx_tr]
        ses_va = sorted(va["sesija_id"].unique().tolist())
        ses_tr = sorted(tr["sesija_id"].unique().tolist())
        raspored = va[TARGET].value_counts().sort_index().to_dict()
        r.p(f"  Fold {i}: validacija sesije {ses_va} ({len(va)} redova) | "
            f"trening sesije {ses_tr} ({len(tr)} redova)")
        r.p(f"           klase u validaciji: {raspored}")
        nedostaju = [k for k in KLASE if raspored.get(k, 0) == 0]
        if nedostaju:
            r.p(f"           UPOZORENJE: nedostaju klase {nedostaju}")

    # Provera da nijedna sesija nije u oba dela istog folda
    for i, (idx_tr, idx_va) in enumerate(foldovi(dev), 1):
        presek = set(dev.iloc[idx_va]["sesija_id"]) & set(dev.iloc[idx_tr]["sesija_id"])
        if presek:
            r.p(f"  GRESKA: fold {i} deli sesije {presek} izmedju treninga i validacije")


def uporedi_modele() -> pd.DataFrame:
    """Faza 6 - pocetno poredjenje modela na development skupu."""
    ensure_dirs()
    r = Izvestaj()

    dev = ucitaj_skup("development")
    atributi = get_features("senzori")

    r.p("POCETNO POREĐENJE MODELA")
    r.p("Faza 6 - osnovne verzije, bez podesavanja hiperparametara")
    r.p("")
    r.p(f"Development skup : {len(dev)} redova, sesije "
        f"{sorted(dev['sesija_id'].unique().tolist())}")
    r.p(f"Test skup        : ZAKLJUCAN, nije ucitan")
    r.p(f"Atributa u modelu: {len(atributi)}")
    r.p(f"  {atributi}")
    r.p("")
    r.p("Vremenski atributi (sat, dan u nedelji...) NISU ukljuceni u ovoj fazi.")
    r.p("Poredjenje sa njima i bez njih je zaseban eksperiment iz specifikacije.")

    r.naslov("1. SASTAV CV FOLDOVA")
    _proveri_foldove(dev, r)

    r.naslov("2. REZULTATI PO MODELIMA")
    r.p("Glavna metrika: macro F1 (svaka klasa nosi istu tezinu).")
    r.p("Prikazane su dve pocetne postavke - podrazumevana i class_weight='balanced'.")

    redovi = []
    oof_po_modelu: dict[str, np.ndarray] = {}

    postavke = [("podrazumevano", False), ("balanced", True)]

    for naziv_postavke, balansirano in postavke:
        modeli = napravi_modele(balansirano)
        if not balansirano:
            modeli = {**napravi_baseline(), **modeli}

        for naziv, (_, treba_tezine) in modeli.items():
            # Model se pravi iznova za svaki fold, da se nista ne prenosi
            def fabrika(n=naziv, b=balansirano):
                svi = {**napravi_baseline(), **napravi_modele(b)}
                return svi[n][0]

            mf1, bacc, acc, oof, t = oceni_model(dev, atributi, fabrika, treba_tezine)
            kljuc = f"{naziv} [{naziv_postavke}]"
            oof_po_modelu[kljuc] = oof
            redovi.append({
                "model": naziv,
                "postavka": naziv_postavke,
                "macroF1_fold1": round(mf1[0], 4),
                "macroF1_fold2": round(mf1[1], 4),
                "macroF1_prosek": round(float(np.mean(mf1)), 4),
                "balanced_acc": round(float(np.mean(bacc)), 4),
                "accuracy": round(float(np.mean(acc)), 4),
                "vreme_s": round(t, 2),
            })

    rez = pd.DataFrame(redovi).sort_values("macroF1_prosek", ascending=False)
    r.p("")
    r.p(rez.to_string(index=False))

    r.p("")
    r.p("Napomena o razlici izmedju foldova: fold 1 validira sesije 4 i 5, a")
    r.p("fold 2 sesije 0, 1 i 7. To su razliciti dani i razlicita osvetljenja,")
    r.p("pa velika razlika izmedju foldova govori da model slabo generalizuje")
    r.p("na novu sesiju - a ne da je CV los.")

    r.naslov("3. METRIKE PO KLASAMA (objedinjene out-of-fold predikcije)")
    r.p("Svaki red developmenta dobio je predikciju modela koji njegovu sesiju")
    r.p("nije video. Dva folda su komplementarna, pa je pokriven ceo skup.")
    y = dev[TARGET].to_numpy()

    for kljuc, oof in oof_po_modelu.items():
        r.p("")
        r.p("-" * 74)
        r.p(kljuc)
        r.p("-" * 74)
        r.p(_tabela_po_klasama(y, oof).to_string())
        r.p(f"  macro F1 = {f1_score(y, oof, average='macro', zero_division=0):.4f}"
            f"   balanced acc = {balanced_accuracy_score(y, oof):.4f}"
            f"   accuracy = {accuracy_score(y, oof):.4f}")

    r.naslov("4. NAPOMENE")
    r.p("- Test skup nije koriscen ni na koji nacin.")
    r.p("- Nije radjeno podesavanje hiperparametara ni izbor atributa.")
    r.p("- Nijedan model jos nije izabran kao konacan.")
    r.p("- StandardScaler je unutar Pipeline-a i fituje se samo na trening delu")
    r.p("  svakog folda, pa statistika validacije ne ucestvuje u skaliranju.")

    # Snimanje
    rez.to_csv(RESULTS_DIR / "07_pocetno_poredjenje_modela.csv", index=False)
    r.snimi(RESULTS_DIR / "07_pocetno_poredjenje_modela.txt")

    nacrtaj_poredjenje(rez)

    print(f"\n\nIzvestaj: {RESULTS_DIR / '07_pocetno_poredjenje_modela.txt'}")
    print(f"Tabela  : {RESULTS_DIR / '07_pocetno_poredjenje_modela.csv'}")
    return rez


def nacrtaj_poredjenje(rez: pd.DataFrame) -> None:
    """Stubicasti grafik macro F1 po modelu, dve postavke jedna uz drugu."""
    modeli = rez["model"].unique().tolist()
    x = np.arange(len(modeli))
    sirina = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor("white")

    for i, postavka in enumerate(["podrazumevano", "balanced"]):
        vrednosti = [
            rez.loc[(rez.model == m) & (rez.postavka == postavka), "macroF1_prosek"]
            .pipe(lambda s: s.iloc[0] if len(s) else np.nan)
            for m in modeli
        ]
        ax.bar(x + (i - 0.5) * sirina, vrednosti, sirina * 0.92,
               label=postavka, color=plt.get_cmap("tab10")(i),
               edgecolor="white", linewidth=1.2, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(modeli, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("macro F1 (prosek 2 folda)", fontsize=9, color="#444444")
    ax.set_title("Pocetno poredjenje modela - development skup, bez tuninga",
                 fontsize=11, color="#222222", loc="left", pad=10)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for strana in ("top", "right"):
        ax.spines[strana].set_visible(False)
    for strana in ("left", "bottom"):
        ax.spines[strana].set_color("#bbbbbb")
    ax.tick_params(colors="#888888", labelsize=8)
    for o in ax.get_xticklabels() + ax.get_yticklabels():
        o.set_color("#444444")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("#444444")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_poredjenje_modela.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ====================================================================
# FAZA 6b - Da li vremenski atributi pomazu generalizaciji?
# ====================================================================
def uporedi_skupove_atributa() -> pd.DataFrame:
    """
    Poredi 'senzori' (16 atributa) i 'senzori_plus_vreme' (16 + 6).

    Pitanje iz specifikacije: da li izvedeni vremenski atributi zaista
    poboljsavaju generalizaciju, ili samo kodiraju raspored eksperimenta?

    Kljucna provera nije prosek, nego SLAGANJE FOLDOVA. Foldovi validiraju
    razlicite sesije (razlicite dane). Ako vremenski atributi pomazu samo u
    jednom foldu, to je znak da model uci raspored, a ne fiziku.
    """
    ensure_dirs()
    r = Izvestaj()

    dev = ucitaj_skup("development")
    skupovi = {"senzori": get_features("senzori"),
               "senzori+vreme": get_features("senzori_plus_vreme")}

    r.p("POREĐENJE: SA VREMENSKIM ATRIBUTIMA I BEZ NJIH")
    r.p("Faza 6b - isti uslovi kao pocetno poredjenje")
    r.p("")
    r.p(f"Development : {len(dev)} redova, sesije "
        f"{sorted(dev['sesija_id'].unique().tolist())}")
    r.p("Test skup   : ZAKLJUCAN, nije ucitan")
    r.p(f"CV          : StratifiedGroupKFold, grupa = sesija_id, {CV_FOLDOVA} folda")
    r.p("")
    for naziv, kolone in skupovi.items():
        r.p(f"  {naziv:14s} ({len(kolone):2d} atributa)")
    dodati = [c for c in skupovi["senzori+vreme"] if c not in skupovi["senzori"]]
    r.p(f"\n  Dodati vremenski atributi: {dodati}")

    r.naslov("1. REZULTATI PO FOLDOVIMA")
    redovi = []
    oof_svi: dict[tuple, np.ndarray] = {}
    y = dev[TARGET].to_numpy()

    for naziv_postavke, balansirano in [("podrazumevano", False), ("balanced", True)]:
        for naziv in napravi_modele(balansirano):
            treba_tezine = napravi_modele(balansirano)[naziv][1]
            for naziv_skupa, kolone in skupovi.items():
                def fabrika(n=naziv, b=balansirano):
                    return napravi_modele(b)[n][0]

                mf1, bacc, acc, oof, t = oceni_model(
                    dev, kolone, fabrika, treba_tezine
                )
                oof_svi[(naziv, naziv_postavke, naziv_skupa)] = oof
                redovi.append({
                    "model": naziv,
                    "postavka": naziv_postavke,
                    "atributi": naziv_skupa,
                    "F1_fold1": round(mf1[0], 4),
                    "F1_fold2": round(mf1[1], 4),
                    "F1_prosek": round(float(np.mean(mf1)), 4),
                    "bal_acc": round(float(np.mean(bacc)), 4),
                    "accuracy": round(float(np.mean(acc)), 4),
                    "vreme_s": round(t, 2),
                })

    rez = pd.DataFrame(redovi)
    r.p(rez.to_string(index=False))

    # ---------------------------------------------------------------- 2
    r.naslov("2. RAZLIKA (senzori+vreme MINUS senzori)")
    r.p("Pozitivna vrednost znaci da su vremenski atributi pomogli.")
    r.p("Kolona 'dosledno' kaze da li su pomogli u OBA folda.")
    r.p("")

    poredjenja = []
    for (model, postavka), deo in rez.groupby(["model", "postavka"], sort=False):
        s = deo[deo["atributi"] == "senzori"].iloc[0]
        v = deo[deo["atributi"] == "senzori+vreme"].iloc[0]
        d1 = v["F1_fold1"] - s["F1_fold1"]
        d2 = v["F1_fold2"] - s["F1_fold2"]
        poredjenja.append({
            "model": model,
            "postavka": postavka,
            "delta_fold1": round(d1, 4),
            "delta_fold2": round(d2, 4),
            "delta_prosek": round(v["F1_prosek"] - s["F1_prosek"], 4),
            "dosledno": "DA" if (d1 > 0 and d2 > 0) else
                        ("NE (oba losija)" if (d1 < 0 and d2 < 0) else "NE (suprotno)"),
        })
    upor = pd.DataFrame(poredjenja)
    r.p(upor.to_string(index=False))

    n_bolje = int((upor["delta_prosek"] > 0).sum())
    n_dosledno = int((upor["dosledno"] == "DA").sum())
    r.p("")
    r.p(f"Vremenski atributi podigli prosek kod {n_bolje} od {len(upor)} kombinacija.")
    r.p(f"Pomogli su u OBA folda kod {n_dosledno} od {len(upor)} kombinacija.")

    # ---------------------------------------------------------------- 3
    r.naslov("3. F1 PO KLASAMA - oba skupa atributa")
    for (model, postavka) in upor[["model", "postavka"]].itertuples(index=False):
        r.p("")
        r.p(f"{model} [{postavka}]")
        tabela = {}
        for naziv_skupa in skupovi:
            oof = oof_svi[(model, postavka, naziv_skupa)]
            _, _, f, _ = precision_recall_fscore_support(
                y, oof, labels=KLASE, zero_division=0
            )
            tabela[naziv_skupa] = f.round(3)
        t = pd.DataFrame(tabela, index=pd.Index(KLASE, name="klasa"))
        t["razlika"] = (t["senzori+vreme"] - t["senzori"]).round(3)
        r.p(t.to_string())

    # ---------------------------------------------------------------- 4
    r.naslov("4. ZAKLJUCAK")
    if n_dosledno >= len(upor) * 0.6:
        r.p("Vremenski atributi dosledno pomazu u oba folda kod vecine modela.")
        r.p("To govori u prilog tome da nose stvarnu informaciju.")
    elif n_bolje <= len(upor) * 0.4:
        r.p("Vremenski atributi uglavnom NE pomazu. Preporuka je zadrzati")
        r.p("skup od 16 senzorskih atributa.")
    else:
        r.p("Rezultat je NEDOSLEDAN: vremenski atributi pomazu u jednom foldu,")
        r.p("a odmazu u drugom. Foldovi validiraju razlicite sesije, pa takav")
        r.p("obrazac odgovara ucenju rasporeda eksperimenta, a ne fizike.")
        r.p("Preporuka je zadrzati samo senzorske atribute.")
    r.p("")
    r.p("NAPOMENA: ovo je nalaz, ne usvojena odluka. Skup atributa se bira")
    r.p("tek posle potvrde.")

    rez.to_csv(RESULTS_DIR / "08_poredjenje_atributa.csv", index=False)
    r.snimi(RESULTS_DIR / "08_poredjenje_atributa.txt")
    nacrtaj_poredjenje_atributa(rez)

    print(f"\n\nIzvestaj: {RESULTS_DIR / '08_poredjenje_atributa.txt'}")
    return rez


def nacrtaj_poredjenje_atributa(rez: pd.DataFrame) -> None:
    """Macro F1 po foldu, dva skupa atributa jedan uz drugi."""
    kombinacije = (
        rez[["model", "postavka"]].drop_duplicates()
        .apply(lambda s: f"{s['model']}\n[{s['postavka']}]", axis=1).tolist()
    )
    parovi = rez[["model", "postavka"]].drop_duplicates().values.tolist()
    x = np.arange(len(parovi))
    sirina = 0.38

    fig, osi = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, fold in zip(osi, ["F1_fold1", "F1_fold2"]):
        for i, naziv_skupa in enumerate(["senzori", "senzori+vreme"]):
            v = [
                rez[(rez.model == m) & (rez.postavka == p) &
                    (rez.atributi == naziv_skupa)][fold].iloc[0]
                for m, p in parovi
            ]
            ax.bar(x + (i - 0.5) * sirina, v, sirina * 0.92, label=naziv_skupa,
                   color=plt.get_cmap("tab10")(i), edgecolor="white",
                   linewidth=1.2, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(kombinacije, fontsize=7.5)
        naslov = ("Fold 1 - validacija sesije 4 i 5" if fold == "F1_fold1"
                  else "Fold 2 - validacija sesije 0, 1 i 7")
        ax.set_title(naslov, fontsize=10, color="#222222", loc="left", pad=8)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbbbbb")
        ax.tick_params(colors="#888888", labelsize=8)
        for o in ax.get_xticklabels() + ax.get_yticklabels():
            o.set_color("#444444")

    osi[0].set_ylabel("macro F1", fontsize=9, color="#444444")
    leg = osi[0].legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("#444444")
    fig.suptitle("Vremenski atributi: pomazu li u OBA folda?",
                 fontsize=12, color="#222222", x=0.007, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_poredjenje_atributa.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ====================================================================
# FAZA 7 - Podesavanje hiperparametara
# ====================================================================
class XGBSaTezinama(XGBClassifier):
    """
    XGBoost koji sam racuna balansirane tezine uzoraka pri fitovanju.

    Zasto ovako: XGBoost nema parametar class_weight. Da bismo balansiranje
    ipak proverili kao opciju, tezine se moraju racunati iz TRENING dela
    svakog folda. Posto se fit() poziva samo nad trening delom, ova klasa
    vidi iskljucivo trening oznake - raspodela validacionog dela ne ucestvuje
    u racunanju tezina.

    NAPOMENA: ovo NIJE class_weight. To je obicno ponderisanje uzoraka i tako
    se i prijavljuje u izvestaju.
    """

    def fit(self, X, y, **kw):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y), **kw)


def mreze_pretrage() -> dict:
    """
    Prostori pretrage. Namerno umereni - dataset je mali, pa ogromna mreza
    ne donosi nista osim vremena.

    class_weight je UKLJUCEN u pretragu (None i 'balanced') tamo gde ga model
    podrzava, jer je pocetno poredjenje pokazalo da ponderisanje ne deluje
    isto na sve modele.
    """
    return {
        "LogisticRegression": {
            "estimator": Pipeline([
                ("skaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            "mreza": {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__class_weight": [None, "balanced"],
            },
            "pretraga": "grid",
        },
        "DecisionTree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "mreza": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 10, 20],
                "min_samples_leaf": [1, 5, 20],
                "criterion": ["gini", "entropy"],
                "class_weight": [None, "balanced"],
            },
            "pretraga": "grid",
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "mreza": {
                "n_estimators": [100, 300],
                "max_depth": [10, None],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 5],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
            },
            "pretraga": "grid",
        },
        # XGBoost: dve odvojene pretrage - bez ponderisanja i sa ponderisanjem
        "XGBoost (bez ponderisanja)": {
            "estimator": XGBClassifier(
                objective="multi:softprob", num_class=len(KLASE),
                eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "mreza": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.3],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "pretraga": "random",
        },
        "XGBoost (sample_weight)": {
            "estimator": XGBSaTezinama(
                objective="multi:softprob", num_class=len(KLASE),
                eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "mreza": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.3],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "pretraga": "random",
        },
    }


def podesi_hiperparametre(naziv_skupa: str = "senzori") -> pd.DataFrame:
    """
    Faza 7 - podesavanje hiperparametara na development skupu.

    Pretraga koristi ISTU usvojenu podelu na foldove (grupa = sesija_id),
    pa nijedan model nikada ne vidi sesiju na kojoj se ocenjuje.
    Kriterijum je macro F1.
    """
    import json
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    ensure_dirs()
    r = Izvestaj()

    dev = ucitaj_skup("development")
    atributi = get_features(naziv_skupa)
    X = dev[atributi].to_numpy()
    y = dev[TARGET].to_numpy()
    cv = foldovi(dev)

    r.p("PODESAVANJE HIPERPARAMETARA")
    r.p("Faza 7 - development skup, zakljucani test se ne koristi")
    r.p("")
    r.p(f"Skup atributa : {naziv_skupa} ({len(atributi)} atributa)")
    r.p(f"Development   : {len(dev)} redova")
    r.p(f"CV            : usvojena podela, grupa = sesija_id, {len(cv)} folda")
    r.p("Kriterijum    : macro F1")
    r.p("")
    r.p("class_weight (None / 'balanced') je deo prostora pretrage kod")
    r.p("LogisticRegression, DecisionTree i RandomForest. XGBoost nema taj")
    r.p("parametar, pa se obican trening i trening sa sample_weight vode kao")
    r.p("dve odvojene pretrage.")

    redovi = []
    najbolji_parametri = {}

    for naziv, spec in mreze_pretrage().items():
        r.p("")
        r.p("=" * 74)
        r.p(naziv)
        r.p("=" * 74)

        if spec["pretraga"] == "grid":
            pretraga = GridSearchCV(
                spec["estimator"], spec["mreza"], scoring="f1_macro",
                cv=cv, n_jobs=-1, refit=False,
            )
            opis_pretrage = "GridSearchCV"
        else:
            pretraga = RandomizedSearchCV(
                spec["estimator"], spec["mreza"], n_iter=25, scoring="f1_macro",
                cv=cv, n_jobs=-1, refit=False, random_state=RANDOM_STATE,
            )
            opis_pretrage = "RandomizedSearchCV (25 kombinacija)"

        pocetak = time.perf_counter()
        pretraga.fit(X, y)
        trajanje = time.perf_counter() - pocetak

        najbolji = pretraga.best_params_
        i = pretraga.best_index_
        f1_po_foldu = [
            round(float(pretraga.cv_results_[f"split{k}_test_score"][i]), 4)
            for k in range(len(cv))
        ]

        r.p(f"Pretraga        : {opis_pretrage}")
        r.p(f"Isprobano kombinacija: {len(pretraga.cv_results_['params'])}")
        r.p(f"Trajanje        : {trajanje:.1f} s")
        r.p("")
        r.p("Najbolji parametri:")
        for kljuc, vrednost in sorted(najbolji.items()):
            r.p(f"  {kljuc:26s} = {vrednost}")
        r.p("")
        r.p(f"macro F1 po foldu : {f1_po_foldu}")
        r.p(f"macro F1 prosek   : {pretraga.best_score_:.4f}")

        # Dodatne metrike za najbolju kombinaciju.
        # best_params_ vec koristi ispravna imena (sa 'model__' prefiksom kod
        # Pipeline-a), pa clone + set_params radi za oba slucaja.
        najbolji_model = clone(spec["estimator"]).set_params(**najbolji)
        _, bacc, acc, oof, _ = oceni_model(
            dev, atributi, lambda m=najbolji_model: clone(m), False
        )
        r.p(f"balanced accuracy : {np.mean(bacc):.4f}")
        r.p(f"accuracy          : {np.mean(acc):.4f}")
        r.p("")
        r.p("Metrike po klasama (objedinjene out-of-fold predikcije):")
        r.p(_tabela_po_klasama(y, oof).to_string())

        najbolji_parametri[naziv] = {str(k): v for k, v in najbolji.items()}
        redovi.append({
            "model": naziv,
            "F1_fold1": f1_po_foldu[0],
            "F1_fold2": f1_po_foldu[1],
            "F1_prosek": round(float(pretraga.best_score_), 4),
            "bal_acc": round(float(np.mean(bacc)), 4),
            "accuracy": round(float(np.mean(acc)), 4),
            "kombinacija": len(pretraga.cv_results_["params"]),
            "vreme_s": round(trajanje, 1),
        })

    rez = pd.DataFrame(redovi).sort_values("F1_prosek", ascending=False)

    r.naslov("ZBIRNA TABELA POSLE PODESAVANJA")
    r.p(rez.to_string(index=False))

    r.naslov("NAPOMENE")
    r.p("- Test skup nije koriscen.")
    r.p("- Izbor atributa nije radjen - to je zasebna faza.")
    r.p("- Konacan model jos nije izabran.")

    with open(MODELS_DIR / "najbolji_parametri.json", "w", encoding="utf-8") as f:
        json.dump(najbolji_parametri, f, indent=2, ensure_ascii=False, default=str)

    rez.to_csv(RESULTS_DIR / "09_hiperparametri.csv", index=False)
    r.snimi(RESULTS_DIR / "09_hiperparametri.txt")

    print(f"\n\nIzvestaj  : {RESULTS_DIR / '09_hiperparametri.txt'}")
    print(f"Parametri : {MODELS_DIR / 'najbolji_parametri.json'}")
    return rez


if __name__ == "__main__":
    uporedi_modele()
    uporedi_skupove_atributa()
    podesi_hiperparametre()

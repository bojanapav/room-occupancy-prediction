"""
FAZA 9 - Konacna evaluacija na zakljucanom test skupu.

Ovo je JEDINO mesto u projektu gde se test skup ucitava.

Postupak, bez odstupanja:
  1. konacni DecisionTree se trenira na CELOM development skupu,
     sa 5 izabranih atributa i hiperparametrima zakljucanim u fazi 7
  2. jednom se izvrsi predikcija nad test skupom (sesije 2 i 6)
  3. rezultat se samo izvestava

Posle ovog koraka se NISTA ne menja na osnovu test rezultata - ni model,
ni atributi, ni hiperparametri, ni podela. Test je procena, ne povratna
informacija za podesavanje.

Pokretanje:
    python src/evaluate.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text

from config import (
    RESULTS_DIR,
    FIGURES_DIR,
    TARGET,
    RANDOM_STATE,
    DT_NAJBOLJI,
    FINALNI_ATRIBUTI,
    ensure_dirs,
)
from data_preparation import Izvestaj
from data_split import ucitaj_skup
from train import oceni_model, KLASE

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

NAZIVI_KLASA = {0: "0 osoba", 1: "1 osoba", 2: "2 osobe", 3: "3 osobe"}


def finalni_model() -> DecisionTreeClassifier:
    """Konacni model - konfiguracija zakljucana pre gledanja testa."""
    return DecisionTreeClassifier(random_state=RANDOM_STATE, **DT_NAJBOLJI)


# ====================================================================
def nacrtaj_matricu(cm: np.ndarray, putanja: str) -> None:
    """Matrica konfuzije: apsolutni brojevi + procenat unutar stvarne klase."""
    cm_proc = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.patch.set_facecolor("white")
    slika = ax.imshow(cm_proc, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(len(KLASE)))
    ax.set_yticks(range(len(KLASE)))
    ax.set_xticklabels([NAZIVI_KLASA[k] for k in KLASE], fontsize=9)
    ax.set_yticklabels([NAZIVI_KLASA[k] for k in KLASE], fontsize=9)

    for i in range(len(KLASE)):
        for j in range(len(KLASE)):
            boja = "white" if cm_proc[i, j] > 55 else "#333333"
            ax.text(j, i, f"{cm[i, j]}\n{cm_proc[i, j]:.1f}%",
                    ha="center", va="center", fontsize=9.5, color=boja)

    ax.set_xlabel("Predvidjena klasa", fontsize=9.5, color="#444444")
    ax.set_ylabel("Stvarna klasa", fontsize=9.5, color="#444444")
    ax.set_title("Matrica konfuzije - test skup (sesije 2 i 6)",
                 fontsize=11, color="#222222", loc="left", pad=10)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="#888888", length=0)
    for o in ax.get_xticklabels() + ax.get_yticklabels():
        o.set_color("#444444")

    traka = fig.colorbar(slika, ax=ax, fraction=0.042, pad=0.03)
    traka.set_label("udeo unutar stvarne klase [%]", fontsize=8.5, color="#444444")
    traka.outline.set_visible(False)
    traka.ax.tick_params(colors="#888888", labelsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / putanja, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def analiza_gresaka(test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                    cm: np.ndarray, r: Izvestaj) -> None:
    """Koje se klase mesaju i sta se iz toga moze zakljuciti."""
    r.naslov("5. ANALIZA GRESAKA")

    # --- najcesce zamene ---
    zamene = []
    for i, stvarna in enumerate(KLASE):
        for j, predvidjena in enumerate(KLASE):
            if i != j and cm[i, j] > 0:
                zamene.append({
                    "stvarna": stvarna,
                    "predvidjena": predvidjena,
                    "broj": int(cm[i, j]),
                    "udeo_klase_%": round(cm[i, j] / cm[i].sum() * 100, 1),
                })
    zamene = pd.DataFrame(zamene).sort_values("broj", ascending=False)

    r.p("A) Najcesce zamene klasa:")
    r.p(zamene.to_string(index=False))

    ukupno_gresaka = int((y_true != y_pred).sum())
    r.p("")
    r.p(f"Ukupno gresaka: {ukupno_gresaka} od {len(y_true)} "
        f"({ukupno_gresaka / len(y_true) * 100:.1f} %)")

    # --- greske po sesiji ---
    r.p("")
    r.p("B) Greske po test sesiji:")
    pomocni = test.assign(tacno=(y_true == y_pred))
    po_sesiji = pomocni.groupby("sesija_id").agg(
        redova=("tacno", "size"),
        gresaka=("tacno", lambda s: int((~s).sum())),
    )
    po_sesiji["stopa_%"] = (po_sesiji["gresaka"] / po_sesiji["redova"] * 100).round(1)
    r.p(po_sesiji.to_string())
    r.p("")
    r.p("  Sesija 2 sadrzi sve cetiri klase; sesija 6 je 24 h prazne prostorije.")

    # --- greske po epizodi ---
    r.p("")
    r.p("C) Greske po epizodi (samo epizode sa bar jednom greskom):")
    po_epizodi = pomocni.groupby("epizoda_id").agg(
        stvarna_klasa=(TARGET, "first"),
        redova=("tacno", "size"),
        gresaka=("tacno", lambda s: int((~s).sum())),
    )
    po_epizodi["stopa_%"] = (po_epizodi["gresaka"] / po_epizodi["redova"] * 100).round(1)
    po_epizodi = po_epizodi[po_epizodi["gresaka"] > 0].sort_values(
        "gresaka", ascending=False
    )
    r.p(po_epizodi.to_string() if len(po_epizodi) else "  Nema gresaka.")
    r.p("")
    r.p("  TUMACENJE: greske su koncentrisane u nekoliko citavih ili gotovo")
    r.p("  citavih epizoda, a ne rasute po celom test skupu. To pokazuje")
    r.p("  vremensku zavisnost uzoraka: kada model pogresi na jednom delu")
    r.p("  epizode, po pravilu gresi i na ostatku iste epizode, jer su")
    r.p("  susedna merenja gotovo identicna. Zbog toga ukupan broj pogresnih")
    r.p("  redova precenjuje koliko je razlicitih situacija model promasio.")

    # --- senzorski profil najcesce zamene ---
    if len(zamene) > 0:
        najveca = zamene.iloc[0]
        s, p = int(najveca["stvarna"]), int(najveca["predvidjena"])
        maska = (y_true == s) & (y_pred == p)
        tacni = (y_true == s) & (y_pred == s)

        r.p("")
        r.p(f"D) Senzorski profil najcesce zamene ({s} -> {p}):")
        kolone = FINALNI_ATRIBUTI + ["S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]
        uporedno = pd.DataFrame({
            f"pogresno ({int(maska.sum())} redova)": test.loc[maska, kolone].median(),
            f"tacno ({int(tacni.sum())} redova)": (
                test.loc[tacni, kolone].median() if tacni.sum() else np.nan
            ),
        })
        r.p(uporedno.to_string(float_format=lambda x: f"{x:9.2f}"))
        r.p("")
        r.p("  Prikazane su medijane. Kolone iznad linije su atributi koje model")
        r.p("  zaista koristi; ispod su CO2 i PIR, koje model NE koristi - stoje")
        r.p("  radi uvida da li je informacija o gresci postojala u drugim senzorima.")


def finalna_evaluacija() -> dict:
    ensure_dirs()
    r = Izvestaj()

    r.p("KONACNA EVALUACIJA NA ZAKLJUCANOM TEST SKUPU")
    r.p("Faza 9 - jedino mesto u projektu gde se test ucitava")

    # ---------------------------------------------------------------- 1
    r.naslov("1. ZAKLJUCANA KONFIGURACIJA")
    r.p("Odluka je doneta iskljucivo na development skupu, pre gledanja testa.")
    r.p("")
    r.p("Model      : DecisionTreeClassifier")
    r.p(f"Atributi   : {len(FINALNI_ATRIBUTI)} - {FINALNI_ATRIBUTI}")
    r.p("Hiperparametri (nadjeni GridSearchCV-om u fazi 7):")
    for k, v in sorted(DT_NAJBOLJI.items()):
        r.p(f"  {k:20s} = {v}")

    dev = ucitaj_skup("development")
    r.p("")
    r.p(f"Trening: CEO development skup, {len(dev)} redova, sesije "
        f"{sorted(dev['sesija_id'].unique().tolist())}")

    # Development CV rezultat iste konfiguracije - za kasnije poredjenje
    mf1_cv, bacc_cv, acc_cv, _, _ = oceni_model(
        dev, FINALNI_ATRIBUTI, finalni_model, False
    )
    cv_rezultat = {
        "macro_F1": float(np.mean(mf1_cv)),
        "balanced_acc": float(np.mean(bacc_cv)),
        "accuracy": float(np.mean(acc_cv)),
        "po_foldu": [round(v, 4) for v in mf1_cv],
    }

    model = finalni_model().fit(dev[FINALNI_ATRIBUTI], dev[TARGET])

    # ---------------------------------------------------------------- 2
    r.naslov("2. TEST SKUP")
    test = ucitaj_skup("test", dozvoli_test=True)   # <-- jedini poziv sa dozvolom
    r.p(f"Redova : {len(test)}")
    r.p(f"Sesije : {sorted(test['sesija_id'].unique().tolist())}")
    r.p(f"Period : {test['Timestamp'].min()}  ->  {test['Timestamp'].max()}")
    r.p("")
    raspodela = test[TARGET].value_counts().sort_index()
    tab = pd.DataFrame({
        "redova": raspodela,
        "procenat_%": (raspodela / len(test) * 100).round(2),
        "epizoda": test.groupby(TARGET)["epizoda_id"].nunique(),
    })
    tab.index.name = "broj_osoba"
    r.p(tab.to_string())

    y_true = test[TARGET].to_numpy()
    y_pred = model.predict(test[FINALNI_ATRIBUTI])

    # ---------------------------------------------------------------- 3
    r.naslov("3. REZULTATI NA TESTU")
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mrec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    r.p(f"Accuracy                      : {acc:.4f}")
    r.p(f"macro F1  (glavna metrika)    : {mf1:.4f}")
    r.p(f"Balanced accuracy             : {bacc:.4f}")
    r.p(f"Macro recall                  : {mrec:.4f}")
    r.p("")
    r.p("  Balanced accuracy i macro recall su ista velicina za viseklasni")
    r.p("  slucaj - prosek odziva po klasama. Navedeni su oba jer ih")
    r.p("  specifikacija trazi pod oba imena.")

    r.naslov("4. METRIKE PO KLASAMA")
    p, rc, f, n = precision_recall_fscore_support(
        y_true, y_pred, labels=KLASE, zero_division=0
    )
    po_klasi = pd.DataFrame(
        {"precision": p.round(3), "recall": rc.round(3),
         "f1": f.round(3), "uzoraka": n},
        index=pd.Index([NAZIVI_KLASA[k] for k in KLASE], name="klasa"),
    )
    r.p(po_klasi.to_string())

    cm = confusion_matrix(y_true, y_pred, labels=KLASE)
    r.p("")
    r.p("Matrica konfuzije (redovi = stvarna klasa, kolone = predvidjena):")
    cm_df = pd.DataFrame(
        cm,
        index=pd.Index([f"stvarno {k}" for k in KLASE]),
        columns=[f"pred. {k}" for k in KLASE],
    )
    r.p(cm_df.to_string())

    nacrtaj_matricu(cm, "12_matrica_konfuzije.png")

    # ---------------------------------------------------------------- 5
    analiza_gresaka(test, y_true, y_pred, cm, r)

    # ---------------------------------------------------------------- 6
    r.naslov("6. POREĐENJE SA DEVELOPMENT CV REZULTATOM")
    r.p("Iskljucivo radi diskusije o generalizaciji. Nista se ne menja.")
    r.p("")
    uporedno = pd.DataFrame({
        "development CV": [round(cv_rezultat["macro_F1"], 4),
                           round(cv_rezultat["balanced_acc"], 4),
                           round(cv_rezultat["accuracy"], 4)],
        "test (sesije 2 i 6)": [round(mf1, 4), round(bacc, 4), round(acc, 4)],
    }, index=["macro F1", "balanced accuracy", "accuracy"])
    uporedno["razlika"] = (uporedno["test (sesije 2 i 6)"]
                           - uporedno["development CV"]).round(4)
    r.p(uporedno.to_string())
    r.p("")
    r.p(f"macro F1 po CV foldovima: {cv_rezultat['po_foldu']}")
    r.p("")
    r.p("Kako citati razliku: test i development se razlikuju i po raspodeli")
    r.p("klasa. Test je 82.8% prazne prostorije, blizu celog skupa, dok")
    r.p("development ima 80.5%. Precision zavisi od te raspodele, pa se")
    r.p("ne prenosi direktno; recall po klasi ne zavisi od prevalence, ali")
    r.p("zavisi od toga koliko su uzorci te klase u testu reprezentativni.")

    # ---------------------------------------------------------------- 7
    r.naslov("7. STRUKTURA KONACNOG STABLA")
    r.p(export_text(model, feature_names=list(FINALNI_ATRIBUTI)))

    r.naslov("8. NAPOMENE")
    r.p("- Test skup je koriscen samo u fazi konacne evaluacije i nikada za")
    r.p("  razvojne odluke.")
    r.p("- Posle gledanja testa nije menjan ni model, ni atributi, ni")
    r.p("  hiperparametri, ni podela.")
    r.p("- Model jos nije eksportovan; deployment je sledeca faza.")

    # Snimanje
    po_klasi.to_csv(RESULTS_DIR / "12_test_po_klasama.csv")
    cm_df.to_csv(RESULTS_DIR / "12_matrica_konfuzije.csv")
    pd.DataFrame([{
        "accuracy": round(acc, 4),
        "macro_F1": round(mf1, 4),
        "balanced_accuracy": round(bacc, 4),
        "macro_recall": round(mrec, 4),
        "cv_macro_F1": round(cv_rezultat["macro_F1"], 4),
        "cv_balanced_accuracy": round(cv_rezultat["balanced_acc"], 4),
        "cv_accuracy": round(cv_rezultat["accuracy"], 4),
    }]).to_csv(RESULTS_DIR / "12_finalna_evaluacija.csv", index=False)
    r.snimi(RESULTS_DIR / "12_finalna_evaluacija.txt")

    print(f"\n\nIzvestaj: {RESULTS_DIR / '12_finalna_evaluacija.txt'}")
    print(f"Grafik  : {FIGURES_DIR / '12_matrica_konfuzije.png'}")
    return {"accuracy": acc, "macro_F1": mf1, "balanced_accuracy": bacc, "cv": cv_rezultat}


if __name__ == "__main__":
    finalna_evaluacija()

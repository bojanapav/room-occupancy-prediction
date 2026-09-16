"""
FAZA 8 - Odabir najznacajnijih atributa.

Pravila koja se postuju:
  - zakljucani test skup (sesije 2 i 6) se NE ucitava
  - radi se iskljucivo na development skupu
  - koristi se ista usvojena 2-fold CV sa sesija_id kao grupom
  - glavna metrika je macro F1
  - polazi se od 16 senzorskih atributa; vremenski atributi NE ucestvuju
    (faza 6b je pokazala da ne poboljsavaju generalizaciju)

Glavni model je DecisionTree sa hiperparametrima nadjenim u fazi 7.

Vazno o racunanju znacaja: znacaj se racuna UNUTAR svakog folda, na trening
delu, a permutaciona provera na validacionom delu tog folda. Nikada se ne
koristi ceo development odjednom, niti test skup.

Pokretanje:
    python src/feature_selection.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

from config import (
    RESULTS_DIR,
    FIGURES_DIR,
    TARGET,
    RANDOM_STATE,
    LIGHT_COLS,
    DT_NAJBOLJI,
    RF_NAJBOLJI,
    get_features,
    ensure_dirs,
)
from data_preparation import Izvestaj
from data_split import ucitaj_skup
from train import foldovi, oceni_model, _tabela_po_klasama, KLASE

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def dt_model() -> DecisionTreeClassifier:
    """DecisionTree sa hiperparametrima iz faze 7."""
    return DecisionTreeClassifier(random_state=RANDOM_STATE, **DT_NAJBOLJI)


def rf_model() -> RandomForestClassifier:
    """RandomForest sa hiperparametrima iz faze 7 - koristi se samo za rangiranje."""
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **RF_NAJBOLJI)


# ====================================================================
# 1. Znacaj atributa
# ====================================================================
def izracunaj_znacaj(dev: pd.DataFrame, atributi: list[str], r: Izvestaj) -> pd.DataFrame:
    """
    Racuna tri mere znacaja, usrednjene preko foldova:

      dt_impurity  - smanjenje necistoce u stablu odlucivanja (trening deo)
      dt_permutac  - pad macro F1 kad se atribut permutuje (validacioni deo)
      rf_impurity  - isto kao prvo, ali iz Random Forest-a

    Zasto i RandomForest: podeseno stablo ima max_depth=3, pa moze da iskoristi
    najvise 7 atributa. Svi ostali dobijaju znacaj tacno 0 i ne mogu se
    medjusobno rangirati. RandomForest koristi sve atribute, pa jedini daje
    potpun poredak nad svih 16.
    """
    X = dev[atributi].to_numpy()
    y = dev[TARGET].to_numpy()

    dt_imp, dt_perm, rf_imp = [], [], []

    for idx_tr, idx_va in foldovi(dev):
        # Stablo odlucivanja - znacaj po necistoci (uci se na treningu)
        dt = dt_model().fit(X[idx_tr], y[idx_tr])
        dt_imp.append(dt.feature_importances_)

        # Permutaciona provera - meri se na VALIDACIONOM delu istog folda
        perm = permutation_importance(
            dt, X[idx_va], y[idx_va], scoring="f1_macro",
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
        )
        dt_perm.append(perm.importances_mean)

        # Random Forest - potpun poredak nad svim atributima
        rf = rf_model().fit(X[idx_tr], y[idx_tr])
        rf_imp.append(rf.feature_importances_)

    znacaj = pd.DataFrame({
        "dt_impurity": np.mean(dt_imp, axis=0).round(4),
        "dt_permutac": np.mean(dt_perm, axis=0).round(4),
        "rf_impurity": np.mean(rf_imp, axis=0).round(4),
    }, index=pd.Index(atributi, name="atribut"))

    r.naslov("1. ZNACAJ ATRIBUTA (usrednjeno preko foldova)")
    r.p("dt_impurity : smanjenje necistoce u podesenom stablu (trening deo folda)")
    r.p("dt_permutac : pad macro F1 kad se atribut permutuje (validacioni deo folda)")
    r.p("rf_impurity : smanjenje necistoce u Random Forest-u (trening deo folda)")
    r.p("")
    r.p(znacaj.sort_values("rf_impurity", ascending=False).to_string())

    nenula = int((znacaj["dt_impurity"] > 0).sum())
    r.p("")
    r.p(f"Atributa sa znacajem vecim od nule u stablu: {nenula} od {len(atributi)}")
    r.p("Podeseno stablo ima max_depth=3, pa fizicki ne moze koristiti vise od 7")
    r.p("atributa. Zato se za rangiranje SVIH 16 koristi rf_impurity, a stablo i")
    r.p("permutaciona mera sluze kao provera slaganja.")

    # Slaganje mera
    r.p("")
    r.p("Slaganje poredaka (Spearman):")
    r.p(f"  rf_impurity vs dt_impurity : "
        f"{znacaj['rf_impurity'].corr(znacaj['dt_impurity'], method='spearman'):.3f}")
    r.p(f"  rf_impurity vs dt_permutac : "
        f"{znacaj['rf_impurity'].corr(znacaj['dt_permutac'], method='spearman'):.3f}")

    return znacaj


def provera_svetlosti(znacaj: pd.DataFrame, poredak: list[str], r: Izvestaj) -> None:
    """Provera hipoteze iz EDA: da li svetlosni atributi dominiraju rangiranjem."""
    r.naslov("2. PROVERA HIPOTEZE O SVETLOSNIM ATRIBUTIMA")
    r.p("U EDA je uocena hipoteza da je svetlost povezana sa sesijom i dobom dana,")
    r.p("a ne samo sa brojem osoba. Ako je tacna, svetlosni atributi bi trebalo da")
    r.p("dominiraju rangiranjem znacaja.")
    r.p("")

    mesta = {a: poredak.index(a) + 1 for a in LIGHT_COLS}
    r.p("Mesto svetlosnih atributa u poretku po rf_impurity:")
    for a, mesto in sorted(mesta.items(), key=lambda p: p[1]):
        r.p(f"  {mesto:2d}. {a:10s}  rf_impurity = {znacaj.loc[a, 'rf_impurity']:.4f}"
            f"   dt_impurity = {znacaj.loc[a, 'dt_impurity']:.4f}")

    udeo = znacaj.loc[LIGHT_COLS, "rf_impurity"].sum() / znacaj["rf_impurity"].sum()
    udeo_dt = (znacaj.loc[LIGHT_COLS, "dt_impurity"].sum()
               / znacaj["dt_impurity"].sum() if znacaj["dt_impurity"].sum() > 0 else 0)
    r.p("")
    r.p(f"Udeo svetlosti u ukupnom znacaju (RandomForest) : {udeo * 100:5.1f} %")
    r.p(f"Udeo svetlosti u ukupnom znacaju (stablo)       : {udeo_dt * 100:5.1f} %")
    r.p(f"Za poredjenje, cetiri od 16 atributa cine        : {4 / 16 * 100:5.1f} %")

    r.p("")
    if udeo > 0.4:
        r.p("NALAZ: svetlosni atributi nose natprosecan deo znacaja.")
    else:
        r.p("NALAZ: svetlosni atributi ne dominiraju rangiranjem.")
    r.p("")
    r.p("Ovo je i dalje samo pokazatelj, ne dokaz. Rangiranje ne razdvaja da li")
    r.p("svetlost nosi informaciju o broju osoba ili o tome koja je sesija u")
    r.p("pitanju. Nijedan atribut se na osnovu ovoga NE uklanja - odluka se")
    r.p("donosi po CV rezultatu u nastavku.")


# ====================================================================
# 3. Sweep po broju atributa
# ====================================================================
def sweep_po_broju(dev: pd.DataFrame, poredak: list[str], r: Izvestaj) -> pd.DataFrame:
    """Za k = 1..16 trenira podeseno stablo na top-k atributa i meri CV."""
    r.naslov("3. PERFORMANSE U ZAVISNOSTI OD BROJA ATRIBUTA")
    r.p("Model: DecisionTree sa hiperparametrima iz faze 7")
    r.p(f"  {DT_NAJBOLJI}")
    r.p("Atributi se dodaju redom po rf_impurity, od najznacajnijeg.")
    r.p("")

    redovi = []
    for k in range(1, len(poredak) + 1):
        izabrani = poredak[:k]
        mf1, bacc, _, _, _ = oceni_model(dev, izabrani, dt_model, False)
        redovi.append({
            "k": k,
            "F1_fold1": round(mf1[0], 4),
            "F1_fold2": round(mf1[1], 4),
            "F1_prosek": round(float(np.mean(mf1)), 4),
            "bal_acc": round(float(np.mean(bacc)), 4),
            "razlika_foldova": round(abs(mf1[0] - mf1[1]), 4),
            "dodat_atribut": izabrani[-1],
        })

    tab = pd.DataFrame(redovi)
    r.p(tab.to_string(index=False))

    r.p("")
    r.p("Atributi po redosledu dodavanja:")
    for i, a in enumerate(poredak, 1):
        r.p(f"  {i:2d}. {a}")

    return tab


def izaberi_skup(tab: pd.DataFrame, poredak: list[str], r: Izvestaj) -> list[str]:
    """
    Bira najmanji smisleni skup atributa.

    Pravilo 1-SE se racuna i prikazuje, ali se NE koristi kao jedini kriterijum:
    sa samo dva folda standardna greska se procenjuje iz dve vrednosti, sto je
    premalo da bi bila pouzdana. Zato se uz nju gleda i stabilnost (razlika
    izmedju foldova).
    """
    r.naslov("4. IZBOR SKUPA ATRIBUTA")

    najbolji_red = tab.loc[tab["F1_prosek"].idxmax()]
    najbolji_k = int(najbolji_red["k"])
    najbolji_f1 = float(najbolji_red["F1_prosek"])

    r.p(f"Najbolji prosecni macro F1: {najbolji_f1:.4f} pri k = {najbolji_k}")

    # 1-SE, samo informativno
    se = tab.loc[tab["k"] == najbolji_k, ["F1_fold1", "F1_fold2"]].values.std(ddof=1)
    se = se / np.sqrt(2)
    prag = najbolji_f1 - se
    kandidati_1se = tab[tab["F1_prosek"] >= prag]["k"].tolist()

    r.p("")
    r.p("Pravilo 1 standardne greske (INFORMATIVNO):")
    r.p(f"  SE procenjena iz 2 folda = {se:.4f}")
    r.p(f"  prag = {najbolji_f1:.4f} - {se:.4f} = {prag:.4f}")
    r.p(f"  k koji zadovoljavaju prag: {kandidati_1se}")
    r.p("")
    r.p("  OGRANICENJE: SE je ovde izracunata iz svega dve vrednosti, pa je")
    r.p("  procena nepouzdana. Zato se 1-SE NE koristi kao jedini kriterijum.")

    # Prakticni kriterijum: najmanji k u granici 2% ispod najboljeg,
    # uz zahtev da razlika izmedju foldova ne bude gora od najboljeg k.
    tolerancija = 0.02
    granica = najbolji_f1 - tolerancija
    najbolja_stabilnost = float(najbolji_red["razlika_foldova"])

    prihvatljivi = tab[
        (tab["F1_prosek"] >= granica)
        & (tab["razlika_foldova"] <= najbolja_stabilnost + 0.05)
    ]
    izabrano_k = int(prihvatljivi["k"].min()) if len(prihvatljivi) else najbolji_k

    r.p("")
    r.p("Primenjeni kriterijum:")
    r.p(f"  najmanji k ciji je prosecni macro F1 unutar {tolerancija:.2f} od najboljeg")
    r.p(f"  ({granica:.4f}) i cija razlika izmedju foldova nije bitno losija")
    r.p(f"  od najboljeg k ({najbolja_stabilnost:.4f} + 0.05)")
    r.p("")
    r.p(f"  IZABRANO: k = {izabrano_k}")

    izabrani = poredak[:izabrano_k]
    r.p("")
    r.p("Izabrani atributi:")
    for i, a in enumerate(izabrani, 1):
        r.p(f"  {i}. {a}")
    return izabrani


# ====================================================================
# 5. Konacno poredjenje
# ====================================================================
def uporedi_svi_vs_izabrani(dev: pd.DataFrame, svi: list[str],
                            izabrani: list[str], r: Izvestaj) -> bool:
    """Poredi podeseno stablo na svim atributima i na izabranom podskupu."""
    r.naslov("5. POREĐENJE: SVI ATRIBUTI vs IZABRANI ATRIBUTI")
    y = dev[TARGET].to_numpy()

    rezultati = {}
    for naziv, kolone in [("svi (16)", svi), (f"izabrani ({len(izabrani)})", izabrani)]:
        mf1, bacc, acc, oof, t = oceni_model(dev, kolone, dt_model, False)
        rezultati[naziv] = {
            "atributa": len(kolone),
            "F1_fold1": round(mf1[0], 4),
            "F1_fold2": round(mf1[1], 4),
            "F1_prosek": round(float(np.mean(mf1)), 4),
            "bal_acc": round(float(np.mean(bacc)), 4),
            "accuracy": round(float(np.mean(acc)), 4),
            "vreme_s": round(t, 3),
            "_oof": oof,
        }

    tab = pd.DataFrame(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
         for k, v in rezultati.items()}
    ).T
    r.p(tab.to_string())

    r.p("")
    r.p("Metrike po klasama:")
    for naziv, v in rezultati.items():
        r.p("")
        r.p(f"{naziv}")
        r.p(_tabela_po_klasama(y, v["_oof"]).to_string())

    kljuc_svi = "svi (16)"
    kljuc_izb = f"izabrani ({len(izabrani)})"
    razlika = rezultati[kljuc_izb]["F1_prosek"] - rezultati[kljuc_svi]["F1_prosek"]

    r.naslov("6. ZAKLJUCAK")
    r.p(f"Razlika u prosecnom macro F1 (izabrani - svi): {razlika:+.4f}")
    r.p("")
    zadrzava = razlika >= -0.02
    if zadrzava:
        r.p(f"Podskup od {len(izabrani)} atributa zadrzava performanse.")
        r.p("=> KANDIDAT ZA KONACAN MODEL.")
        r.p("")
        r.p("Sto se tice fizickog sistema: ako manji broj senzorskih velicina")
        r.p("daje prakticno isti rezultat, sistem se moze pojednostaviti.")
    else:
        r.p(f"Podskup od {len(izabrani)} atributa znacajno gubi performanse.")
        r.p("=> ZADRZAVAMO SVIH 16 ATRIBUTA.")

    r.p("")
    r.p("NAPOMENA: test skup nije koriscen. Konacan model nije izabran niti")
    r.p("eksportovan - to je sledeca faza, posle potvrde.")
    return zadrzava


def nacrtaj_sweep(tab: pd.DataFrame, izabrano_k: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.patch.set_facecolor("white")

    ax.plot(tab["k"], tab["F1_fold1"], marker="o", markersize=4, linewidth=1.2,
            color=plt.get_cmap("tab10")(0), label="fold 1 (val: sesije 4, 5)")
    ax.plot(tab["k"], tab["F1_fold2"], marker="o", markersize=4, linewidth=1.2,
            color=plt.get_cmap("tab10")(1), label="fold 2 (val: sesije 0, 1, 7)")
    ax.plot(tab["k"], tab["F1_prosek"], marker="s", markersize=5, linewidth=2,
            color=plt.get_cmap("tab10")(2), label="prosek")

    ax.axvline(izabrano_k, color="#888888", linewidth=1)
    ax.annotate(f"izabrano k = {izabrano_k}", xy=(izabrano_k, 0.05),
                xytext=(izabrano_k + 0.3, 0.05), fontsize=9, color="#444444")

    ax.set_xticks(tab["k"])
    ax.set_xlabel("Broj atributa (top-k po rf_impurity)", fontsize=9, color="#444444")
    ax.set_ylabel("macro F1", fontsize=9, color="#444444")
    ax.set_title("Performanse u zavisnosti od broja atributa (development CV)",
                 fontsize=11, color="#222222", loc="left", pad=10)
    ax.set_ylim(0, 1)
    ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors="#888888", labelsize=8)
    for o in ax.get_xticklabels() + ax.get_yticklabels():
        o.set_color("#444444")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color("#444444")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_broj_atributa.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def nacrtaj_znacaj(znacaj: pd.DataFrame) -> None:
    z = znacaj.sort_values("rf_impurity")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor("white")
    boje = [plt.get_cmap("tab10")(1) if a in LIGHT_COLS else plt.get_cmap("tab10")(0)
            for a in z.index]
    ax.barh(z.index, z["rf_impurity"], color=boje, edgecolor="white",
            linewidth=1.0, height=0.7, zorder=2)
    ax.set_xlabel("rf_impurity (prosek preko foldova)", fontsize=9, color="#444444")
    ax.set_title("Znacaj atributa - svetlosni atributi istaknuti",
                 fontsize=11, color="#222222", loc="left", pad=10)
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors="#888888", labelsize=8)
    for o in ax.get_xticklabels() + ax.get_yticklabels():
        o.set_color("#444444")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_znacaj_atributa.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ====================================================================
def pokreni_izbor_atributa() -> list[str]:
    ensure_dirs()
    r = Izvestaj()

    dev = ucitaj_skup("development")
    atributi = get_features("senzori")

    r.p("ODABIR NAJZNACAJNIJIH ATRIBUTA")
    r.p("Faza 8 - development skup; zakljucani test se ne ucitava")
    r.p("")
    r.p(f"Development     : {len(dev)} redova, sesije "
        f"{sorted(dev['sesija_id'].unique().tolist())}")
    r.p(f"Pocetni atributi: {len(atributi)} senzorskih")
    r.p("Vremenski atributi NE ucestvuju (faza 6b: ne poboljsavaju generalizaciju)")
    r.p(f"CV              : usvojena podela, grupa = sesija_id, 2 folda")

    znacaj = izracunaj_znacaj(dev, atributi, r)
    poredak = znacaj.sort_values("rf_impurity", ascending=False).index.tolist()

    provera_svetlosti(znacaj, poredak, r)
    tab = sweep_po_broju(dev, poredak, r)
    izabrani = izaberi_skup(tab, poredak, r)
    zadrzava = uporedi_svi_vs_izabrani(dev, atributi, izabrani, r)

    # Prikaz samog stabla - korisno za dokumentaciju i odbranu
    r.naslov("7. STRUKTURA STABLA NA IZABRANIM ATRIBUTIMA")
    r.p("Stablo je fitovano na CELOM development skupu samo radi prikaza")
    r.p("pravila. Ocene performansi iznad dolaze iskljucivo iz CV-a.")
    r.p("")
    stablo = dt_model().fit(dev[izabrani], dev[TARGET])
    r.p(export_text(stablo, feature_names=list(izabrani)))

    znacaj.to_csv(RESULTS_DIR / "10_znacaj_atributa.csv")
    tab.to_csv(RESULTS_DIR / "10_broj_atributa.csv", index=False)
    r.snimi(RESULTS_DIR / "10_izbor_atributa.txt")

    nacrtaj_sweep(tab, len(izabrani))
    nacrtaj_znacaj(znacaj)

    print(f"\n\nIzvestaj: {RESULTS_DIR / '10_izbor_atributa.txt'}")
    print(f"Izabrani atributi ({len(izabrani)}): {izabrani}")
    print(f"Podskup zadrzava performanse: {zadrzava}")
    return izabrani


# ====================================================================
# Dijagnosticka provera: da li je svetlost informacija ili precica?
# ====================================================================
def dijagnostika_svetlosti(izabrani: list[str] | None = None) -> pd.DataFrame:
    """
    Kontrolisano poredjenje istog podesenog stabla na tri skupa atributa:

      A) svih 16 senzorskih atributa
      B) skup izabran u ovoj fazi
      C) senzorski atributi BEZ cetiri Light atributa

    Menja se ISKLJUCIVO skup atributa. Hiperparametri, podela, foldovi i
    metrika ostaju identicni, pa je razlika u rezultatu posledica atributa,
    a ne podesavanja.

    Pitanje: da li veliko oslanjanje na svetlost nosi korisnu informaciju,
    ili je precica vezana za sesije snimanja?
    """
    ensure_dirs()
    r = Izvestaj()

    dev = ucitaj_skup("development")
    svi = get_features("senzori")
    if izabrani is None:
        izabrani = ["S2_Light", "S1_Light", "S3_Light", "S4_Light", "S2_Temp"]
    bez_svetla = [c for c in svi if c not in LIGHT_COLS]

    skupovi = {
        f"A) svi senzori ({len(svi)})": svi,
        f"B) izabrani ({len(izabrani)})": izabrani,
        f"C) bez svetlosti ({len(bez_svetla)})": bez_svetla,
    }

    r.p("DIJAGNOSTICKA PROVERA: SVETLOST - INFORMACIJA ILI PRECICA?")
    r.p("Dodatak fazi 8; nije nova faza projekta")
    r.p("")
    r.p(f"Development : {len(dev)} redova, sesije "
        f"{sorted(dev['sesija_id'].unique().tolist())}")
    r.p("Test skup   : ZAKLJUCAN, nije ucitan")
    r.p("Model       : DecisionTree sa hiperparametrima iz faze 7, NEPROMENJEN")
    r.p(f"  {DT_NAJBOLJI}")
    r.p("CV          : usvojena podela, grupa = sesija_id, 2 folda")
    r.p("")
    r.p("Menja se samo skup atributa. Hiperparametri se NE podesavaju posebno")
    r.p("ni za jednu varijantu - to bi pokvarilo kontrolisano poredjenje.")
    r.p("")
    for naziv, kolone in skupovi.items():
        r.p(f"  {naziv}")
        r.p(f"     {kolone}")

    y = dev[TARGET].to_numpy()
    redovi, oof_svi = [], {}

    for naziv, kolone in skupovi.items():
        mf1, bacc, acc, oof, _ = oceni_model(dev, kolone, dt_model, False)
        oof_svi[naziv] = oof
        redovi.append({
            "skup": naziv,
            "atributa": len(kolone),
            "F1_fold1": round(mf1[0], 4),
            "F1_fold2": round(mf1[1], 4),
            "F1_prosek": round(float(np.mean(mf1)), 4),
            "razlika_foldova": round(abs(mf1[0] - mf1[1]), 4),
            "bal_acc": round(float(np.mean(bacc)), 4),
            "accuracy": round(float(np.mean(acc)), 4),
        })

    tab = pd.DataFrame(redovi)

    r.naslov("1. ZBIRNI REZULTATI")
    r.p(tab.to_string(index=False))

    r.naslov("2. METRIKE PO KLASAMA")
    for naziv, oof in oof_svi.items():
        r.p("")
        r.p("-" * 74)
        r.p(naziv)
        r.p("-" * 74)
        r.p(_tabela_po_klasama(y, oof).to_string())

    r.naslov("3. KLASA 3 - POSEBNO")
    r.p("Klasa 3 je bila najslabija u svim dosadasnjim rezultatima, a upravo su")
    r.p("njene epizode u development skupu snimane po mraku. Ako je svetlost")
    r.p("precica, njeno uklanjanje bi ovde trebalo da napravi najvecu razliku.")
    r.p("")
    kl3 = {}
    for naziv, oof in oof_svi.items():
        t = _tabela_po_klasama(y, oof).loc[3]
        kl3[naziv] = {"precision": t["precision"], "recall": t["recall"], "f1": t["f1"]}
    r.p(pd.DataFrame(kl3).T.to_string())

    r.p("")
    r.p("F1 po klasama, sva tri skupa:")
    poredjenje = pd.DataFrame(
        {naziv: _tabela_po_klasama(y, oof)["f1"] for naziv, oof in oof_svi.items()}
    )
    r.p(poredjenje.to_string())

    # ---------------------------------------------------------------- 4
    r.naslov("4. TUMACENJE")
    najbolji = tab["F1_prosek"].max()
    red_c = tab[tab["skup"].str.startswith("C")].iloc[0]
    pad = najbolji - red_c["F1_prosek"]
    stabilniji = red_c["razlika_foldova"] <= tab["razlika_foldova"].min() + 1e-9

    r.p(f"Najbolji prosecni macro F1 medju skupovima : {najbolji:.4f}")
    r.p(f"Skup bez svetlosti                         : {red_c['F1_prosek']:.4f}")
    r.p(f"Pad u odnosu na najbolji                   : {pad:.4f}")
    r.p(f"Razlika izmedju foldova (bez svetlosti)    : {red_c['razlika_foldova']:.4f}")
    r.p(f"Najmanja razlika izmedju foldova ima skup  : "
        f"{tab.loc[tab['razlika_foldova'].idxmin(), 'skup']}")
    r.p("")

    if pad <= 0.03 and stabilniji:
        r.p("NALAZ: skup bez svetlosti drzi se blizu najboljeg rezultata i pri tome")
        r.p("je stabilniji izmedju foldova. To ide u prilog tome da je oslanjanje")
        r.p("na svetlost velikim delom precica vezana za sesije snimanja.")
        r.p("=> Fizicki smisleniji skup vredi razmotriti kao kandidata.")
    elif pad <= 0.03:
        r.p("NALAZ: skup bez svetlosti drzi se blizu najboljeg rezultata, ali nije")
        r.p("stabilniji izmedju foldova. Rezultat je nedovoljno jasan da bi sam")
        r.p("po sebi opravdao uklanjanje svetlosti.")
    elif pad >= 0.10:
        r.p("NALAZ: uklanjanje svetlosti znacajno obara rezultat. To govori da")
        r.p("svetlosni atributi nose vaznu prediktivnu informaciju u ovom skupu")
        r.p("podataka, bez obzira na povezanost sa sesijom.")
        r.p("=> Zadrzavamo zakljucak da je svetlost korisna.")
    else:
        r.p("NALAZ: rezultat je izmedju - pad postoji, ali nije drastican.")
        r.p("Odluka se ne moze doneti samo na osnovu ovog broja; treba pogledati")
        r.p("i ponasanje po klasama, posebno klasu 3.")

    r.p("")
    r.p("OGRANICENJE OVE PROVERE: dva folda i mali broj epizoda znace da su sve")
    r.p("razlike merene sa velikom nesigurnoscu. Nalaz je pokazatelj, ne dokaz.")
    r.p("")
    r.p("NAPOMENA: test skup nije koriscen. Konacan skup atributa i konacan")
    r.p("model nisu izabrani.")

    tab.to_csv(RESULTS_DIR / "11_dijagnostika_svetlosti.csv", index=False)
    r.snimi(RESULTS_DIR / "11_dijagnostika_svetlosti.txt")

    print(f"\n\nIzvestaj: {RESULTS_DIR / '11_dijagnostika_svetlosti.txt'}")
    return tab


if __name__ == "__main__":
    izabrani = pokreni_izbor_atributa()
    dijagnostika_svetlosti(izabrani)

"""
FAZA 4 - Eksplorativna analiza (EDA).

VAZNO: analiza se radi ISKLJUCIVO na development skupu. Test skup (sesije 2 i 6)
je zakljucan u fazi 5 i ovde se ni ne ucitava - nijedna odluka o podacima,
atributima ili modelima ne sme se doneti gledajuci ga.

Ovde se NE radi feature selection, treniranje ni tuning. Cilj je razumeti
podatke i uociti sta trazi uputstvo projekta:
  - raspodele numerickih atributa
  - anomalije i ekstremne vrednosti
  - korelacije izmedju atributa, posebno unutar grupa senzora
  - odnos S5_CO2 i S5_CO2_Slope
  - ponasanje PIR senzora
  - razlike atributa izmedju klasa 0, 1, 2, 3
  - potencijalno redundantni atributi

Pokretanje:
    python src/eda.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # bez GUI-ja, samo snimanje u fajl
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR,
    FIGURES_DIR,
    TARGET,
    SENSOR_COLS,
    TEMP_COLS,
    LIGHT_COLS,
    SOUND_COLS,
    PIR_COLS,
    ensure_dirs,
)
from data_preparation import Izvestaj

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

# ====================================================================
# Boje - standardne matplotlib vrednosti, birane samo radi citljivosti
# ====================================================================
SURFACE = "white"
INK = "#222222"       # naslovi
INK2 = "#444444"      # oznake osa
MUTED = "#888888"     # crtice na osama
GRID = "#dddddd"      # mreza
AXIS = "#bbbbbb"      # osnovna linija

# Broj osoba je uredjena velicina (0 < 1 < 2 < 3), pa klase koriste
# matplotlib rampu "Blues" - tamnije znaci vise osoba.
KLASA_BOJE = [plt.get_cmap("Blues")(v) for v in (0.35, 0.55, 0.72, 0.90)]

# Vise serija na istom grafiku - standardna tab10 paleta
SERIJA = [plt.get_cmap("tab10")(i) for i in range(4)]

# Korelacije idu od -1 do +1, pa koristimo standardnu divergentnu mapu
CMAP_KOREL = "RdBu_r"

PRAG_REDUNDANSE = 0.90  # |r| iznad ovoga = kandidat za redundantan par


def _stil(ax, naslov: str = "", x: str = "", y: str = "") -> None:
    """Zajednicki stil: tanke marke, recesivna mreza, bez suvisnih linija."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for strana in ("top", "right"):
        ax.spines[strana].set_visible(False)
    for strana in ("left", "bottom"):
        ax.spines[strana].set_color(AXIS)
        ax.spines[strana].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for oznaka in ax.get_xticklabels() + ax.get_yticklabels():
        oznaka.set_color(INK2)
    if naslov:
        ax.set_title(naslov, color=INK, fontsize=10, pad=8, loc="left")
    if x:
        ax.set_xlabel(x, color=INK2, fontsize=9)
    if y:
        ax.set_ylabel(y, color=INK2, fontsize=9)


def _figura(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _snimi(fig, naziv: str) -> str:
    putanja = FIGURES_DIR / naziv
    fig.savefig(putanja, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return naziv


# ====================================================================
# Ucitavanje - samo development
# ====================================================================
def ucitaj_development() -> pd.DataFrame:
    """
    Vraca ISKLJUCIVO development redove.

    Koristi zajednicki ucitavac iz data_split, koji odbija da ucita test skup
    bez eksplicitne dozvole - pa ovaj modul fizicki ne moze da ga vidi.
    """
    from data_split import ucitaj_skup

    return ucitaj_skup("development")


# ====================================================================
# 1. Raspodela klasa
# ====================================================================
def fig_raspodela_klasa(dev: pd.DataFrame, r: Izvestaj) -> None:
    r.naslov("1. RASPODELA KLASA U DEVELOPMENT SKUPU")
    broj = dev[TARGET].value_counts().sort_index()
    proc = (broj / len(dev) * 100).round(2)
    epiz = dev.groupby(TARGET)["epizoda_id"].nunique()
    tab = pd.DataFrame({"redova": broj, "procenat_%": proc, "epizoda": epiz})
    tab.index.name = "broj_osoba"
    r.p(tab.to_string())
    r.p("")
    r.p("ZAKLJUCAK: prazna soba cini vecinu razvojnog skupa. Zbog toga tacnost")
    r.p("nije dovoljna metrika - glavni kriterijum je macro F1, uz balanced")
    r.p("accuracy i metrike po klasama.")

    fig, ax = _figura(figsize=(6.2, 3.6))
    stubici = ax.bar(broj.index.astype(str), broj.values,
                     color=[KLASA_BOJE[k] for k in broj.index],
                     edgecolor=SURFACE, linewidth=1.5, width=0.62, zorder=2)
    # Direktne oznake samo na vrhu stubica - ne na svakoj tacki podataka
    for s, v, p in zip(stubici, broj.values, proc.values):
        ax.text(s.get_x() + s.get_width() / 2, v + len(dev) * 0.015,
                f"{v}\n{p:.1f}%", ha="center", va="bottom",
                color=INK2, fontsize=8.5)
    ax.set_ylim(0, broj.max() * 1.20)
    _stil(ax, "Raspodela broja osoba (development skup)", "Broj osoba u prostoriji", "Broj merenja")
    r.p(f"\n[grafik] {_snimi(fig, '01_raspodela_klasa.png')}")


# ====================================================================
# 2. Raspodele numerickih atributa
# ====================================================================
def fig_raspodele_atributa(dev: pd.DataFrame, r: Izvestaj) -> None:
    r.naslov("2. RASPODELE NUMERICKIH ATRIBUTA")
    opis = dev[SENSOR_COLS].describe().T
    opis["nunique"] = dev[SENSOR_COLS].nunique()
    opis["asimetrija"] = dev[SENSOR_COLS].skew().round(2)
    r.p(opis.to_string(float_format=lambda x: f"{x:11.3f}"))

    r.p("")
    r.p("ZAPAZANJA:")
    for c in LIGHT_COLS:
        udeo_nula = (dev[c] == 0).mean() * 100
        r.p(f"  {c:10s}: {udeo_nula:5.1f}% merenja je tacno 0 (mrak)")
    r.p("  => Senzori svetlosti imaju veliku masu u nuli; raspodela je")
    r.p("     dvomodalna (mrak / upaljeno svetlo), a ne zvonasta.")
    r.p("")
    for c in SOUND_COLS:
        r.p(f"  {c:10s}: medijana {dev[c].median():.2f}, maksimum {dev[c].max():.2f}"
            f"  (odnos max/medijana = {dev[c].max() / dev[c].median():.0f}x)")
    r.p("  => Zvuk je izrazito desno asimetrican: tisina uz retke kratke skokove.")

    fig, osi = plt.subplots(4, 4, figsize=(13, 9))
    fig.patch.set_facecolor(SURFACE)
    for ax, c in zip(osi.ravel(), SENSOR_COLS):
        ax.hist(dev[c], bins=40, color=SERIJA[0], edgecolor=SURFACE,
                linewidth=0.4, zorder=2)
        _stil(ax, c)
        ax.tick_params(labelsize=7)
    fig.suptitle("Raspodele senzorskih atributa (development skup)",
                 color=INK, fontsize=12, x=0.077, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    r.p(f"\n[grafik] {_snimi(fig, '02_raspodele_atributa.png')}")


# ====================================================================
# 3. Anomalije i ekstremne vrednosti
# ====================================================================
def analiza_anomalija(dev: pd.DataFrame, r: Izvestaj) -> None:
    r.naslov("3. ANOMALIJE, EKSTREMNE I CUDNE VREDNOSTI")

    r.p("A) Temperature - poredjenje cetiri senzora:")
    t = dev[TEMP_COLS].describe().T[["min", "50%", "max", "std"]]
    t["opseg"] = (t["max"] - t["min"]).round(2)
    r.p(t.to_string(float_format=lambda x: f"{x:8.2f}"))

    najveci = t["max"].idxmax()
    ostali_max = t.drop(index=najveci)["max"].max()
    r.p("")
    r.p(f"  {najveci} ima maksimum {t.loc[najveci, 'max']:.2f} C, dok nijedan drugi")
    r.p(f"  temperaturni senzor ne prelazi {ostali_max:.2f} C.")
    r.p("  Moguce objasnjenje: taj cvor je blize izvoru toplote ili osobama.")
    r.p("  Vrednost je fizicki moguca za zatvorenu prostoriju, pa se NE uklanja.")

    r.p("\nB) Broj ekstrema po IQR pravilu (informativno, nista se ne brise):")
    for c in SENSOR_COLS:
        q1, q3 = dev[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            n = int(((dev[c] < q1) | (dev[c] > q3)).sum())
            dodatak = "  (IQR=0, raspodela koncentrisana u jednoj vrednosti)"
        else:
            n = int(((dev[c] < q1 - 1.5 * iqr) | (dev[c] > q3 + 1.5 * iqr)).sum())
            dodatak = ""
        r.p(f"  {c:14s} {n:5d}  ({n / len(dev) * 100:5.2f} %){dodatak}")

    r.p("")
    r.p("  VAZNO: kod senzora ekstremna vrednost najcesce znaci da se dogadjaj")
    r.p("  stvarno desio (upaljeno svetlo, ulazak osobe, zvuk). To je signal koji")
    r.p("  model treba da nauci, a ne sum - zato se ekstremi zadrzavaju.")

    r.p("\nC) Kratke epizode (moguce nepreciznosti u obelezavanju):")
    kratke = dev.groupby("epizoda_id").agg(
        klasa=(TARGET, "first"), redova=(TARGET, "size")
    )
    kratke = kratke[kratke["redova"] <= 5].sort_values("redova")
    if len(kratke) == 0:
        r.p("  Nema epizoda krace od 6 merenja.")
    else:
        r.p(kratke.to_string())
        r.p(f"\n  Ukupno {len(kratke)} epizoda traje najvise 5 merenja (~2.5 min).")
        r.p("  To su verovatno trenuci ulaska/izlaska. NE uklanjaju se bez dogovora -")
        r.p("  ako se ukloni epizoda, menja se i podela podataka.")


# ====================================================================
# 4. Korelacije izmedju atributa
# ====================================================================
def fig_korelacije(dev: pd.DataFrame, r: Izvestaj) -> pd.DataFrame:
    r.naslov("4. KORELACIJE IZMEDJU ATRIBUTA")
    kor = dev[SENSOR_COLS].corr()

    fig, ax = _figura(figsize=(9.5, 8))
    slika = ax.imshow(kor.values, cmap=CMAP_KOREL, vmin=-1, vmax=1)
    ax.set_xticks(range(len(SENSOR_COLS)))
    ax.set_yticks(range(len(SENSOR_COLS)))
    ax.set_xticklabels(SENSOR_COLS, rotation=90, fontsize=8)
    ax.set_yticklabels(SENSOR_COLS, fontsize=8)
    for i in range(len(SENSOR_COLS)):
        for j in range(len(SENSOR_COLS)):
            v = kor.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.4,
                    color="#ffffff" if abs(v) > 0.55 else INK2)
    ax.grid(False)
    for strana in ax.spines.values():
        strana.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    for o in ax.get_xticklabels() + ax.get_yticklabels():
        o.set_color(INK2)
    ax.set_title("Pearsonova korelacija senzorskih atributa (development skup)",
                 color=INK, fontsize=11, pad=10, loc="left")
    traka = fig.colorbar(slika, ax=ax, fraction=0.035, pad=0.02)
    traka.outline.set_visible(False)
    traka.ax.tick_params(colors=MUTED, labelsize=8)
    r.p(f"[grafik] {_snimi(fig, '03_korelaciona_matrica.png')}")

    # Najjaci parovi
    parovi = (
        kor.where(np.triu(np.ones(kor.shape), k=1).astype(bool))
        .stack()
        .rename("r")
        .reset_index()
        .rename(columns={"level_0": "atribut_A", "level_1": "atribut_B"})
    )
    parovi["|r|"] = parovi["r"].abs()
    parovi = parovi.sort_values("|r|", ascending=False)

    r.p("\n15 najjacih korelacija:")
    r.p(parovi.head(15).to_string(index=False, float_format=lambda x: f"{x:6.3f}"))
    return parovi


# ====================================================================
# 5. Korelacije unutar grupa senzora
# ====================================================================
def fig_korelacije_grupa(dev: pd.DataFrame, r: Izvestaj) -> None:
    r.naslov("5. KORELACIJE UNUTAR GRUPA SENZORA")
    r.p("Pitanje: da li cetiri senzora iste vrste mere prakticno istu stvar?")

    grupe = {"Temperatura": TEMP_COLS, "Svetlost": LIGHT_COLS, "Zvuk": SOUND_COLS}

    fig, osi = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.patch.set_facecolor(SURFACE)

    for ax, (naziv, kolone) in zip(osi, grupe.items()):
        k = dev[kolone].corr()
        vandijag = k.values[np.triu_indices_from(k.values, k=1)]
        r.p("")
        r.p(f"{naziv} ({', '.join(kolone)}):")
        r.p(k.to_string(float_format=lambda x: f"{x:6.3f}"))
        r.p(f"  prosecna korelacija van dijagonale: {vandijag.mean():.3f}")
        r.p(f"  opseg: {vandijag.min():.3f} do {vandijag.max():.3f}")

        ax.imshow(k.values, cmap=CMAP_KOREL, vmin=-1, vmax=1)
        ax.set_xticks(range(len(kolone)))
        ax.set_yticks(range(len(kolone)))
        ax.set_xticklabels([c.split("_")[0] for c in kolone], fontsize=8)
        ax.set_yticklabels([c.split("_")[0] for c in kolone], fontsize=8)
        for i in range(len(kolone)):
            for j in range(len(kolone)):
                v = k.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(v) > 0.55 else INK2)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
        for o in ax.get_xticklabels() + ax.get_yticklabels():
            o.set_color(INK2)
        ax.set_title(f"{naziv}  (prosek r = {vandijag.mean():.2f})",
                     color=INK, fontsize=10, pad=8, loc="left")

    fig.tight_layout()
    r.p(f"\n[grafik] {_snimi(fig, '04_korelacije_grupa.png')}")


# ====================================================================
# 6. CO2 i CO2_Slope
# ====================================================================
def fig_co2(dev: pd.DataFrame, r: Izvestaj) -> None:
    r.naslov("6. ODNOS S5_CO2 I S5_CO2_Slope")
    rp = dev["S5_CO2"].corr(dev["S5_CO2_Slope"])
    r.p(f"Pearsonova korelacija: r = {rp:.3f}")
    r.p("")
    r.p("S5_CO2_Slope je brzina promene koncentracije, dakle izvod S5_CO2 po")
    r.p("vremenu. Nivo i njegov izvod nose razlicitu informaciju: nivo govori")
    r.p("koliko se CO2 nakupilo, a nagib da li se bas sada nakuplja ili opada.")
    r.p("Niska korelacija potvrdjuje da NISU redundantni.")

    r.p("\nPo klasama:")
    po_klasi = dev.groupby(TARGET).agg(
        CO2_medijana=("S5_CO2", "median"),
        CO2_max=("S5_CO2", "max"),
        Slope_medijana=("S5_CO2_Slope", "median"),
        Slope_prosek=("S5_CO2_Slope", "mean"),
    ).round(3)
    r.p(po_klasi.to_string())
    r.p("")
    r.p("ZAPAZANJE: u development skupu medijane i S5_CO2 i S5_CO2_Slope")
    r.p("monotono rastu kroz klase 0 -> 1 -> 2 -> 3. To je opis ovog skupa")
    r.p("podataka, ne tvrdnja o ponasanju senzora uopste.")

    # Small multiples po klasi - izbegava preklapanje cetiri boje u jednom oblaku
    fig, osi = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, k in zip(osi, [0, 1, 2, 3]):
        pod = dev[dev[TARGET] == k]
        ax.scatter(pod["S5_CO2"], pod["S5_CO2_Slope"], s=9, alpha=0.45,
                   color=KLASA_BOJE[k], linewidths=0, zorder=2)
        _stil(ax, f"{k} osoba  (n={len(pod)})", "S5_CO2 [ppm]",
              "S5_CO2_Slope" if k == 0 else "")
    fig.suptitle("CO2 nivo naspram brzine promene, po broju osoba",
                 color=INK, fontsize=11, x=0.062, ha="left", y=1.02)
    fig.tight_layout()
    r.p(f"\n[grafik] {_snimi(fig, '05_co2_i_nagib.png')}")


# ====================================================================
# 7. PIR senzori
# ====================================================================
def fig_pir(dev: pd.DataFrame, r: Izvestaj) -> None:
    r.naslov("7. PONASANJE PIR SENZORA")
    r.p("PIR detektuje pokret, ne prisustvo. Osoba koja mirno sedi ga ne aktivira.")
    r.p("")

    tab = dev.groupby(TARGET)[PIR_COLS].mean().mul(100).round(1)
    tab.columns = [f"{c}_aktivan_%" for c in PIR_COLS]
    tab["bar_jedan_%"] = (
        dev.assign(bilo_koji=dev[PIR_COLS].sum(axis=1) > 0)
        .groupby(TARGET)["bilo_koji"].mean().mul(100).round(1)
    )
    tab.index.name = "broj_osoba"
    r.p(tab.to_string())

    slaganje = (dev["S6_PIR"] == dev["S7_PIR"]).mean() * 100
    r.p(f"\nS6_PIR i S7_PIR daju istu vrednost u {slaganje:.1f}% merenja.")
    r.p(f"Korelacija izmedju njih: r = {dev['S6_PIR'].corr(dev['S7_PIR']):.3f}")

    prazno_aktivan = dev.loc[dev[TARGET] == 0, PIR_COLS].sum(axis=1).gt(0).mean() * 100
    zauzeto_neaktivan = dev.loc[dev[TARGET] > 0, PIR_COLS].sum(axis=1).eq(0).mean() * 100
    r.p("")
    r.p(f"U samo {prazno_aktivan:.1f}% merenja prazne prostorije bio je aktivan")
    r.p("bar jedan PIR senzor. Drugim recima, PIR pokazuje veoma visoku")
    r.p("specificnost za praznu prostoriju - lazna detekcija je retka.")
    r.p("")
    r.p(f"Obrnuto, u {zauzeto_neaktivan:.1f}% merenja sa bar jednom osobom oba PIR")
    r.p("senzora su bila mirna. PIR reaguje na pokret, ne na prisustvo, pa osoba")
    r.p("koja mirno sedi ostaje nezapazena.")
    r.p("")
    r.p("ZAKLJUCAK: PIR je pouzdan pokazatelj da NEKO jeste u sobi, ali sam ne")
    r.p("moze da razlikuje 1, 2 ili 3 osobe. Ocekujemo da modelu bude koristan")
    r.p("za odvajanje prazno/zauzeto, a ne za brojanje.")

    fig, ax = _figura(figsize=(6.6, 3.6))
    x = np.arange(4)
    sirina = 0.36
    for i, c in enumerate(PIR_COLS):
        v = dev.groupby(TARGET)[c].mean().mul(100)
        ax.bar(x + (i - 0.5) * sirina, v.values, sirina * 0.94, label=c,
               color=SERIJA[i], edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(4)])
    _stil(ax, "Udeo merenja sa aktivnim PIR senzorom, po broju osoba",
          "Broj osoba u prostoriji", "Aktivan [%]")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    r.p(f"\n[grafik] {_snimi(fig, '06_pir_po_klasama.png')}")


# ====================================================================
# 8. Razlike atributa izmedju klasa
# ====================================================================
def fig_razlike_po_klasama(dev: pd.DataFrame, r: Izvestaj) -> None:
    r.naslov("8. RAZLIKE ATRIBUTA IZMEDJU KLASA 0, 1, 2, 3")

    r.p("Medijana svakog atributa po klasi:")
    med = dev.groupby(TARGET)[SENSOR_COLS].median().T
    r.p(med.to_string(float_format=lambda x: f"{x:9.3f}"))

    # Opisna mera razdvojivosti: udeo varijanse objasnjen pripadnoscu klasi.
    # Ovo je SAMO opis za EDA - izbor atributa se radi kasnije, propisanim
    # metodama i iskljucivo na trening podacima.
    r.p("")
    r.p("Opisna mera razdvojivosti (udeo varijanse objasnjen klasom, eta^2):")
    r.p("NAPOMENA: ovo je samo opis podataka za EDA, NE izbor atributa.")
    redovi = []
    for c in SENSOR_COLS:
        ukupna = ((dev[c] - dev[c].mean()) ** 2).sum()
        grupe = dev.groupby(TARGET)[c]
        izmedju = (grupe.count() * (grupe.mean() - dev[c].mean()) ** 2).sum()
        redovi.append({"atribut": c, "eta2": izmedju / ukupna if ukupna > 0 else 0.0})
    eta = pd.DataFrame(redovi).sort_values("eta2", ascending=False).set_index("atribut")
    r.p(eta.to_string(float_format=lambda x: f"{x:7.4f}"))

    # ---- zapazanje o svetlosti, formulisano kao HIPOTEZA ----
    r.p("")
    r.p("ZAPAZANJE: medijane svetlosti nisu monotone po broju osoba.")
    med_l = dev.groupby(TARGET)[LIGHT_COLS].median()
    r.p(med_l.T.to_string(float_format=lambda x: f"{x:8.1f}"))
    r.p("")
    r.p("Kod vise senzora klasa 3 ima NIZU medijanu svetlosti od klasa 1 i 2.")
    r.p("")
    r.p("HIPOTEZA (nije potvrdjena i ne menja preprocesiranje):")
    r.p("  Moguce objasnjenje je da su epizode klase 3 u razvojnom skupu")
    r.p("  snimane kasno popodne i uvece, pa je izmerena svetlost povezana sa")
    r.p("  sesijom i dobom dana, a ne samo sa brojem osoba. Ako je tako, visok")
    r.p("  eta^2 kod svetlosti delom odrazava koja je epizoda u pitanju.")
    r.p("")
    r.p("  Ovo je zasad SAMO pretpostavka. Proverava se kasnije - poredjenjem")
    r.p("  modela sa vremenskim atributima i bez njih, vaznoscu atributa i")
    r.p("  analizom gresaka. Nijedan atribut se na osnovu ovoga ne uklanja.")

    fig, osi = plt.subplots(4, 4, figsize=(13, 9))
    fig.patch.set_facecolor(SURFACE)
    for ax, c in zip(osi.ravel(), SENSOR_COLS):
        podaci = [dev.loc[dev[TARGET] == k, c].values for k in range(4)]
        bp = ax.boxplot(podaci, patch_artist=True, widths=0.55,
                        medianprops=dict(color=SURFACE, linewidth=1.4),
                        whiskerprops=dict(color=AXIS, linewidth=0.9),
                        capprops=dict(color=AXIS, linewidth=0.9),
                        flierprops=dict(marker="o", markersize=1.8,
                                        markerfacecolor=MUTED,
                                        markeredgecolor="none", alpha=0.35))
        for zakrpa, k in zip(bp["boxes"], range(4)):
            zakrpa.set_facecolor(KLASA_BOJE[k])
            zakrpa.set_edgecolor(SURFACE)
            zakrpa.set_linewidth(1.2)
        ax.set_xticklabels(["0", "1", "2", "3"])
        _stil(ax, c)
        ax.tick_params(labelsize=7)
    fig.suptitle("Raspodela atributa po broju osoba (development skup)",
                 color=INK, fontsize=12, x=0.077, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    r.p(f"\n[grafik] {_snimi(fig, '07_atributi_po_klasama.png')}")


# ====================================================================
# 9. Kandidati za redundantne atribute
# ====================================================================
def redundantni_atributi(parovi: pd.DataFrame, r: Izvestaj) -> None:
    r.naslov("9. POTENCIJALNO REDUNDANTNI ATRIBUTI")
    r.p(f"Kriterijum: |r| >= {PRAG_REDUNDANSE}")
    r.p("")
    jaki = parovi[parovi["|r|"] >= PRAG_REDUNDANSE]
    if len(jaki) == 0:
        r.p("Nijedan par atributa ne prelazi prag.")
    else:
        r.p(jaki.to_string(index=False, float_format=lambda x: f"{x:6.3f}"))
    r.p("")
    r.p("VAZNO: ovo je samo SPISAK KANDIDATA, ne odluka. Nijedan atribut se")
    r.p("sada ne uklanja. Odabir najznacajnijih atributa je posebna faza")
    r.p("projekta i radi se propisanim metodama, na trening podacima.")


# ====================================================================
def pokreni_edu() -> None:
    """Faza 4 - kompletna EDA na development skupu."""
    ensure_dirs()
    r = Izvestaj()

    dev = ucitaj_development()

    r.p("EKSPLORATIVNA ANALIZA (EDA)")
    r.p("Faza 4 - iskljucivo na DEVELOPMENT skupu; test se ne ucitava")
    r.p("")
    r.p(f"Redova u analizi : {len(dev)}")
    r.p(f"Sesije           : {sorted(dev['sesija_id'].unique().tolist())}")
    r.p(f"Epizoda          : {dev['epizoda_id'].nunique()}")
    r.p(f"Atributa (senzori): {len(SENSOR_COLS)}")

    fig_raspodela_klasa(dev, r)
    fig_raspodele_atributa(dev, r)
    analiza_anomalija(dev, r)
    parovi = fig_korelacije(dev, r)
    fig_korelacije_grupa(dev, r)
    fig_co2(dev, r)
    fig_pir(dev, r)
    fig_razlike_po_klasama(dev, r)
    redundantni_atributi(parovi, r)

    izlaz = RESULTS_DIR / "06_eda_izvestaj.txt"
    r.snimi(izlaz)
    print(f"\n\nIzvestaj snimljen u: {izlaz}")
    print(f"Grafici snimljeni u: {FIGURES_DIR}")


if __name__ == "__main__":
    pokreni_edu()

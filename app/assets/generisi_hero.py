"""
Generise hero sliku za Streamlit aplikaciju.

Slika je originalna i crta se programski, pa nema ni eksternog URL-a ni
pitanja licence. Pokrece se jednom; rezultat (hero.png) se cuva u repou.

Pokretanje:
    python app/assets/generisi_hero.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

IZLAZ = Path(__file__).resolve().parent / "hero.png"

# Paleta - tamnoplava ka tirkiznoj
TAMNA = "#0d2137"
SREDNJA = "#13415a"
TIRKIZ = "#2fb8a8"
SVETLO_PLAVA = "#5aa9d6"
LED = "#7ee8d5"


def zaobljena(ax, x, y, w, h, boja, alpha=1.0, r=0.06, ivica="none", lw=0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=boja, alpha=alpha, edgecolor=ivica, linewidth=lw, zorder=3,
    ))


def osoba(ax, x, y, visina, boja, alpha=1.0):
    """Ravna figura: glava + ramena + telo, sa uzim proporcijama."""
    r = visina * 0.145
    vrat = visina * 0.055
    ax.add_patch(Circle((x, y + visina - r), r, facecolor=boja,
                        alpha=alpha, edgecolor="none", zorder=5))
    telo_v = visina - 2 * r - vrat
    sirina = visina * 0.30
    ax.add_patch(FancyBboxPatch(
        (x - sirina / 2, y), sirina, telo_v,
        boxstyle=f"round,pad=0,rounding_size={sirina * 0.32}",
        facecolor=boja, alpha=alpha, edgecolor="none", zorder=5,
    ))


def senzor(ax, x, y_plafon, dubina, boja_snopa=LED):
    """Senzor na plafonu sa snopom detekcije."""
    sirina_snopa = dubina * 0.62
    # zorder 3.6: iznad zida (3), ispod ljudi (5)
    ax.add_patch(Polygon(
        [(x, y_plafon), (x - sirina_snopa, y_plafon - dubina),
         (x + sirina_snopa, y_plafon - dubina)],
        closed=True, facecolor=boja_snopa, alpha=0.16,
        edgecolor="none", zorder=3.6,
    ))
    zaobljena(ax, x - 0.19, y_plafon - 0.10, 0.38, 0.13, "#dbe7f0", r=0.05)
    ax.add_patch(Circle((x, y_plafon - 0.035), 0.045,
                        facecolor=LED, edgecolor="none", zorder=6))


def napravi_hero() -> Path:
    fig = plt.figure(figsize=(16, 5.4), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # ---- pozadina: dijagonalni gradijent tamnoplava -> tirkizna ----
    n = 400
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    polje = np.clip(gx * 0.78 + (1 - gy) * 0.30, 0, 1)
    mapa = matplotlib.colors.LinearSegmentedColormap.from_list(
        "pozadina", [TAMNA, SREDNJA, "#155e63"]
    )
    ax.imshow(polje, extent=(0, 16, 0, 5.4), cmap=mapa,
              origin="lower", aspect="auto", zorder=0)

    # ---- diskretna tackasta mreza (tehnicki utisak) ----
    rng = np.random.default_rng(7)
    tx = rng.uniform(0, 16, 260)
    ty = rng.uniform(0, 5.4, 260)
    ax.scatter(tx, ty, s=2.2, color="#ffffff", alpha=0.07, zorder=1)

    # ===================== SCENA (desna polovina platna) =====================
    POD = 1.30
    PLAFON = 4.60
    LEVO, DESNO = 7.9, 15.5

    # zadnji zid
    zaobljena(ax, LEVO, POD, DESNO - LEVO, PLAFON - POD, "#0a2233",
              alpha=0.92, r=0.14)

    # prozori
    for i in range(3):
        x0 = LEVO + 0.45 + i * 2.45
        zaobljena(ax, x0, 2.55, 1.95, 1.62, "#1a6981", alpha=0.9, r=0.08)
        zaobljena(ax, x0 + 0.07, 2.63, 0.82, 1.46, "#3ba2bd", alpha=0.45, r=0.05)

    # linije plafona i poda
    ax.plot([LEVO, DESNO], [PLAFON, PLAFON], color="#ffffff",
            alpha=0.16, lw=1.4, zorder=4)
    ax.plot([LEVO - 0.3, DESNO + 0.2], [POD, POD], color=LED,
            alpha=0.30, lw=1.8, zorder=4)

    # senzori na plafonu sa snopovima
    for sx in (9.4, 11.7, 14.0):
        senzor(ax, sx, PLAFON - 0.03, PLAFON - POD - 0.20)

    # stolovi i monitori
    for dx in (8.55, 10.95, 13.35):
        zaobljena(ax, dx, POD + 0.34, 1.75, 0.13, "#35839b", alpha=0.95, r=0.05)
        for nx in (dx + 0.16, dx + 1.59):
            ax.plot([nx, nx], [POD, POD + 0.34], color="#35839b",
                    lw=2.6, alpha=0.85, zorder=4)
        zaobljena(ax, dx + 0.52, POD + 0.47, 0.72, 0.48, "#0d3445",
                  alpha=1.0, r=0.04)
        zaobljena(ax, dx + 0.57, POD + 0.52, 0.62, 0.38, "#4fb6cf",
                  alpha=0.75, r=0.03)

    # ljudi
    osoba(ax, 9.40, POD + 0.02, 1.42, LED, alpha=0.97)
    osoba(ax, 10.35, POD + 0.02, 1.24, SVETLO_PLAVA, alpha=0.9)
    osoba(ax, 11.70, POD + 0.02, 1.46, TIRKIZ, alpha=0.97)
    osoba(ax, 14.00, POD + 0.02, 1.33, SVETLO_PLAVA, alpha=0.88)

    # oznake detekcije iznad glava
    for px, py in ((9.40, 3.05), (11.70, 3.10), (14.00, 2.95)):
        ax.add_patch(Circle((px, py), 0.13, facecolor="none",
                            edgecolor=LED, lw=1.6, alpha=0.8, zorder=6))
        ax.add_patch(Circle((px, py), 0.05, facecolor=LED,
                            edgecolor="none", alpha=1.0, zorder=6))

    # ---- traka senzorskih velicina, ispod scene ----
    kartice = [
        (8.05, "SVETLOST", "#f0c05a"),
        (9.95, "TEMPERATURA", "#f08a6a"),
        (12.20, "CO₂", "#6ad4b0"),
        (13.75, "POKRET", "#9aa8f0"),
    ]
    for kx, tekst, boja in kartice:
        sirina = 0.62 + len(tekst) * 0.105
        zaobljena(ax, kx, 0.40, sirina, 0.44, "#ffffff", alpha=0.09, r=0.10)
        ax.add_patch(Circle((kx + 0.24, 0.62), 0.085, facecolor=boja,
                            edgecolor="none", zorder=6))
        ax.text(kx + 0.42, 0.62, tekst, color="#ffffff", alpha=0.78,
                fontsize=8.5, va="center", ha="left", zorder=6,
                fontfamily="DejaVu Sans")

    fig.savefig(IZLAZ, dpi=100, facecolor=TAMNA)
    plt.close(fig)
    return IZLAZ


if __name__ == "__main__":
    p = napravi_hero()
    print(f"Snimljeno: {p}  ({p.stat().st_size / 1024:.0f} KB)")

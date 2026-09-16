"""
Glavni pokretac projekta.

`python pipeline.py` izvrsava ceo proces ispravnim redosledom, tako da se svi
rezultati mogu reprodukovati jednom komandom.

Napomena o redosledu: podela podataka se radi PRE eksplorativne analize.
Razlog je sto svaka odluka doneta gledanjem odnosa atributa i ciljne
promenljive na celom skupu vec predstavlja oblik curenja informacija. Zato se
podela prvo zakljuca, a EDA se zatim radi iskljucivo na development skupu.

Pokretanje:
    python pipeline.py              # svi koraci redom
    python pipeline.py --korak 3    # samo treci korak
    python pipeline.py --spisak     # ispis koraka
"""

import argparse
import sys
from pathlib import Path

# Omogucava `from config import ...` unutar modula u src/
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))


def korak_pregled_podataka() -> None:
    """Ucitavanje sirovog CSV-a i izvestaj o kvalitetu podataka."""
    from data_preparation import inspect_raw

    inspect_raw()


def korak_preprocesiranje() -> None:
    """Enkodiranje vremena i priprema obradjenog skupa."""
    from data_preparation import build_processed

    build_processed()


def korak_struktura() -> None:
    """Analiza epizoda zauzetosti - podloga za odluku o podeli."""
    from data_split import analiza_strukture

    analiza_strukture()


def korak_podela() -> None:
    """Finalna podela na development/test i provera CV foldova."""
    from data_split import finalna_podela, predlozi_cv_foldove

    finalna_podela()
    predlozi_cv_foldove()


def korak_eda() -> None:
    """Eksplorativna analiza - iskljucivo na development skupu."""
    from eda import pokreni_edu

    pokreni_edu()


def korak_treniranje() -> None:
    """Pocetno treniranje cetiri modela + baseline, bez tuninga."""
    from train import uporedi_modele

    uporedi_modele()


def korak_poredjenje_atributa() -> None:
    """Provera da li izvedeni vremenski atributi pomazu generalizaciji."""
    from train import uporedi_skupove_atributa

    uporedi_skupove_atributa()


def korak_hiperparametri() -> None:
    """Podesavanje hiperparametara na development skupu."""
    from train import podesi_hiperparametre

    podesi_hiperparametre()


def korak_izbor_atributa() -> None:
    """
    Odabir najznacajnijih atributa na development skupu, uz dijagnosticku
    proveru koliko rezultat zavisi od svetlosnih atributa.
    """
    from feature_selection import pokreni_izbor_atributa, dijagnostika_svetlosti

    izabrani = pokreni_izbor_atributa()
    dijagnostika_svetlosti(izabrani)


def korak_evaluacija() -> None:
    """Konacna evaluacija - JEDINO mesto gde se test skup ucitava."""
    from evaluate import finalna_evaluacija

    finalna_evaluacija()


def korak_eksport() -> None:
    """Finalno treniranje na developmentu i eksport modela u models/."""
    from export_model import eksportuj_model

    eksportuj_model()


# Koraci u REDOSLEDU IZVRSAVANJA. Novi koraci se dopisuju na kraj liste.
# NAPOMENA: Streamlit aplikacija NIJE deo pipeline-a. Ona samo ucitava
# eksportovani model i pokrece se posebno:  streamlit run app/app.py
KORACI = [
    ("Ucitavanje i pregled dataseta", korak_pregled_podataka),
    ("Preprocesiranje i enkodiranje", korak_preprocesiranje),
    ("Analiza epizoda zauzetosti", korak_struktura),
    ("Finalna podela podataka i CV foldovi", korak_podela),
    ("Eksplorativna analiza (samo development skup)", korak_eda),
    ("Pocetno treniranje i poredjenje modela", korak_treniranje),
    ("Poredjenje: sa vremenskim atributima i bez njih", korak_poredjenje_atributa),
    ("Podesavanje hiperparametara", korak_hiperparametri),
    ("Odabir najznacajnijih atributa", korak_izbor_atributa),
    ("Konacna evaluacija na zakljucanom test skupu", korak_evaluacija),
    ("Eksport konacnog modela (deployment)", korak_eksport),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Room Occupancy - pipeline")
    parser.add_argument(
        "--korak",
        type=int,
        choices=range(1, len(KORACI) + 1),
        help="Pokrece samo jedan korak. Bez ovog argumenta pokrecu se svi.",
    )
    parser.add_argument(
        "--spisak", action="store_true", help="Ispisuje spisak koraka i izlazi."
    )
    args = parser.parse_args()

    if args.spisak:
        for i, (opis, _) in enumerate(KORACI, 1):
            print(f"  {i}. {opis}")
        return

    za_izvrsavanje = [args.korak] if args.korak else range(1, len(KORACI) + 1)

    for broj in za_izvrsavanje:
        opis, funkcija = KORACI[broj - 1]
        print(f"\n{'#' * 78}")
        print(f"# KORAK {broj}/{len(KORACI)}: {opis}")
        print(f"{'#' * 78}")
        funkcija()


if __name__ == "__main__":
    main()

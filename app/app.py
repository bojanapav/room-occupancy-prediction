"""
Streamlit aplikacija - predikcija broja osoba u prostoriji.

Aplikacija UCITAVA vec istrenirani model iz models/finalni_model.joblib.
Model se ne trenira pri pokretanju.

Pokretanje iz korena projekta:
    streamlit run app/app.py
"""

import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

KOREN = Path(__file__).resolve().parents[1]
MODEL_FAJL = KOREN / "models" / "finalni_model.joblib"
HERO_SLIKA = Path(__file__).resolve().parent / "assets" / "hero.png"

# Prikazni nazivi. Kljucevi su STVARNA imena atributa i ne smeju se menjati -
# menja se samo tekst koji korisnik vidi.
NAZIVI = {
    "S1_Light": ("Svetlost — senzor S1", "lux"),
    "S2_Light": ("Svetlost — senzor S2", "lux"),
    "S3_Light": ("Svetlost — senzor S3", "lux"),
    "S4_Light": ("Svetlost — senzor S4", "lux"),
    "S1_Temp": ("Temperatura — senzor S1", "°C"),
    "S2_Temp": ("Temperatura — senzor S2", "°C"),
    "S3_Temp": ("Temperatura — senzor S3", "°C"),
    "S4_Temp": ("Temperatura — senzor S4", "°C"),
}

# Redosled u kojem se polja PRIKAZUJU. Nema veze sa redosledom koji ide
# modelu - taj se uvek uzima iz paketa.
REDOSLED_PRIKAZA = ["S1_Light", "S2_Light", "S3_Light", "S4_Light", "S2_Temp"]

OPIS_KLASE = {0: "prostorija je prazna", 1: "osoba u prostoriji",
              2: "osobe u prostoriji", 3: "osobe u prostoriji"}


@st.cache_resource
def ucitaj_model():
    """Ucitava eksportovani paket. cache_resource -> ucitava se samo jednom."""
    if not MODEL_FAJL.exists():
        return None
    return joblib.load(MODEL_FAJL)


@st.cache_data
def hero_base64() -> str:
    """Ucitava lokalnu hero sliku i kodira je za CSS pozadinu."""
    if not HERO_SLIKA.exists():
        return ""
    return base64.b64encode(HERO_SLIKA.read_bytes()).decode()


st.set_page_config(
    page_title="Predikcija broja osoba u prostoriji",
    page_icon="👥",
    layout="centered",
)

# ------------------------------------------------------------------- stil
st.markdown(
    """
<style>
  .stApp { background: #f4f7fb; }
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 880px; }

  .hero-okvir {
      position: relative; border-radius: 18px; overflow: hidden;
      min-height: 230px; display: flex; align-items: center;
      box-shadow: 0 12px 34px rgba(13, 33, 55, 0.28);
      margin-bottom: 26px;
  }
  .hero-tekst { position: relative; padding: 30px 34px; max-width: 62%; }
  .hero-tekst h1 {
      font-size: 1.92rem; font-weight: 700; color: #ffffff;
      margin: 0 0 8px 0; line-height: 1.2; letter-spacing: -0.4px;
  }
  .hero-tekst p {
      margin: 0; font-size: 1.02rem; color: rgba(255,255,255,0.88);
  }
  .hero-tekst .predmet {
      margin-top: 16px; font-size: 0.78rem; letter-spacing: 1.1px;
      color: rgba(255,255,255,0.6); text-transform: uppercase;
  }

  .kartica {
      background: #ffffff; border: 1px solid #e4e9f2; border-radius: 16px;
      padding: 22px 26px 8px 26px;
      box-shadow: 0 3px 14px rgba(16, 32, 56, 0.06);
      margin-bottom: 20px;
  }
  .kartica-naslov {
      font-size: 1.06rem; font-weight: 650; color: #0f2740; margin: 0 0 4px 0;
  }
  .kartica-opis { font-size: 0.85rem; color: #6b7c93; margin: 0 0 6px 0; }

  .rezultat {
      background: linear-gradient(135deg, #0d2137 0%, #13566a 55%, #17897f 100%);
      border-radius: 18px; padding: 30px 34px; text-align: center;
      color: #ffffff; margin: 8px 0 20px 0;
      box-shadow: 0 12px 30px rgba(13, 33, 55, 0.26);
  }
  .rezultat .oznaka {
      font-size: 0.8rem; letter-spacing: 1.4px; text-transform: uppercase;
      color: rgba(255,255,255,0.65); margin-bottom: 4px;
  }
  .rezultat .broj {
      font-size: 5.2rem; font-weight: 700; line-height: 1;
      color: #7ee8d5; margin: 6px 0 2px 0;
  }
  .rezultat .opis { font-size: 1.02rem; color: rgba(255,255,255,0.9); }

  .red-ver {
      display: flex; justify-content: space-between; font-size: 0.85rem;
      color: #33445c; margin-top: 13px; margin-bottom: 4px;
  }
  .red-ver .vred { color: #7b8ba3; font-variant-numeric: tabular-nums; }
  .traka { background: #eaeff6; border-radius: 5px; height: 8px; overflow: hidden; }
  .traka > div { background: linear-gradient(90deg,#13566a,#17897f); height: 100%; }

  div.stButton > button {
      width: 100%; height: 3.1rem; border-radius: 12px;
      font-weight: 650; font-size: 1.02rem; letter-spacing: 0.2px;
      background: linear-gradient(135deg, #13566a 0%, #17897f 100%) !important;
      color: #ffffff !important; border: none !important;
      box-shadow: 0 6px 16px rgba(19, 86, 106, 0.28);
      transition: transform .08s ease, box-shadow .15s ease;
  }
  div.stButton > button:hover {
      box-shadow: 0 8px 20px rgba(19, 86, 106, 0.36);
      transform: translateY(-1px);
  }
  div.stButton > button:focus { box-shadow: 0 0 0 3px rgba(23,137,127,0.3) !important; }

  div[data-testid="stExpander"] {
      border: 1px solid #e4e9f2; border-radius: 12px; background: #ffffff;
  }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------- hero
slika = hero_base64()
pozadina = (
    f"background-image:url('data:image/png;base64,{slika}');"
    "background-size:cover;background-position:center right;"
) if slika else "background:linear-gradient(135deg,#0d2137,#17897f);"

st.markdown(
    f"""
<div class="hero-okvir" style="{pozadina}">
  <div class="hero-tekst">
    <h1>Predikcija broja osoba u prostoriji</h1>
    <p>Procena zauzetosti na osnovu senzorskih podataka</p>
    <div class="predmet">SAUSAU · Room Occupancy Estimation</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

paket = ucitaj_model()

if paket is None:
    st.error(
        f"Model nije pronađen na putanji: {MODEL_FAJL}\n\n"
        "Pokreni prvo: `python pipeline.py`"
    )
    st.stop()

atributi = paket["atributi"]          # REDOSLED KOJI IDE MODELU - ne dirati
opsezi = paket["opsezi"]

# Redosled samo za prikaz; sve sto nije nabrojano ide na kraj.
prikaz = ([a for a in REDOSLED_PRIKAZA if a in atributi]
          + [a for a in atributi if a not in REDOSLED_PRIKAZA])

# -------------------------------------------------------------- unos
st.markdown(
    f"""
<div class="kartica" style="padding-bottom:2px">
  <div class="kartica-naslov">Očitavanja senzora</div>
  <div class="kartica-opis">
    Model koristi {len(atributi)} atributa izabrana u fazi odabira najznačajnijih
    atributa. Granice polja odgovaraju opsegu vrednosti iz trening podataka.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

vrednosti = {}
kolone = st.columns(2)
for i, atribut in enumerate(prikaz):
    o = opsezi[atribut]
    je_ceo_broj = o["decimala"] == 0
    naziv, jedinica = NAZIVI.get(atribut, (atribut, ""))
    with kolone[i % 2]:
        vrednosti[atribut] = st.number_input(
            label=f"{naziv}  ·  {jedinica}" if jedinica else naziv,
            min_value=float(o["min"]),
            max_value=float(o["max"]),
            value=float(o["medijana"]),
            step=1.0 if je_ceo_broj else 0.01,
            format="%.0f" if je_ceo_broj else "%.2f",
            help=f"Interni naziv atributa: {atribut}. "
                 f"Opseg u trening podacima: {o['min']:g} – {o['max']:g}",
        )

st.write("")

# -------------------------------------------------------- predikcija
if st.button("Predvidi broj osoba", type="primary"):
    # Redosled kolona uzimamo iz paketa, nikada ga ne pretpostavljamo rucno.
    red = pd.DataFrame([[vrednosti[a] for a in atributi]], columns=atributi)
    predikcija = int(paket["model"].predict(red)[0])

    st.markdown(
        f"""
<div class="rezultat">
  <div class="oznaka">Predviđeni broj osoba</div>
  <div class="broj">{predikcija}</div>
  <div class="opis">{OPIS_KLASE.get(predikcija, "")}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    verovatnoce = paket["model"].predict_proba(red)[0]
    trake = "".join(
        f"""<div class="red-ver">
              <span>{k} {"osoba" if k in (0, 1) else "osobe"}</span>
              <span class="vred">{v:.1%}</span>
            </div>
            <div class="traka"><div style="width:{v * 100:.1f}%"></div></div>"""
        for k, v in zip(paket["klase"], verovatnoce)
    )
    st.markdown(
        f"""
<div class="kartica" style="padding-bottom:22px">
  <div class="kartica-naslov">Verovatnoće koje model dodeljuje klasama</div>
  <div class="kartica-opis">
    Vrednosti koje model vraća za svaku klasu. Nisu kalibrisana mera sigurnosti —
    kod stabla odlučivanja odgovaraju raspodeli klasa u listu u koji uzorak pada.
  </div>
  {trake}
</div>
""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ info
st.write("")
with st.expander("O modelu"):
    meta = paket["metapodaci"]
    st.markdown(
        f"""
Aplikacija procenjuje **broj osoba u prostoriji** na osnovu senzorskih očitavanja.
Moguće klase su **0, 1, 2 i 3 osobe**.

- **Tip modela:** {meta['tip_modela']}
- **Trenirano na:** {meta['trenirano_na']}, {meta['broj_redova_treninga']} merenja
- **Skaliranje:** {meta['skaliranje']}
- **Datum eksporta:** {meta['datum_eksporta']}
- **Redosled ulaznih atributa:** `{', '.join(atributi)}`

Ovo je demonstracija već istreniranog modela — aplikacija ga samo učitava iz
`models/` i ne trenira ga pri pokretanju.
"""
    )
    st.write("**Hiperparametri:**")
    st.json(paket["hiperparametri"])

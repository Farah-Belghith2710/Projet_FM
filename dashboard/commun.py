"""Outils communs au dashboard : chargement des résultats, couleurs, style des pages et des graphiques."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))          # pour « from src import economie » dans les pages
RESULTATS = RACINE / "results"
TRAITE = RACINE / "data" / "processed"

# ---------------------------------------------------------------- couleurs
# Thème sombre : fond noir bleuté, accents bleus. Les modèles gardent les mêmes teintes que dans les
# figures du rapport, avec des nuances un peu ajustées pour rester lisibles sur fond noir.
ENCRE, ENCRE_2, DISCRET = "#E6EBF3", "#AEB8C8", "#8A94A6"   # texte principal, secondaire, discret
GRILLE, AXE, FOND = "#1D2533", "#2F3B4F", "#10151F"         # grille, axes, fond des graphiques
NEUTRE = "#4E596B"                                         # éléments « en retrait » (gris)
COULEURS_MODELES = {
    "Fiabilité (B10)": DISCRET, "Wiener": "#3987e5", "SVR": "#d95926", "GPR": "#199e70",
    "Random Forest": "#c98500", "LSTM": "#d55181", "CNN 1D": "#008300",
}
CATEGORIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
COULEURS_CONDITIONS = {1: "#256abf", 2: "#5598e7", 3: "#b7d3f6"}   # charge plus forte = bleu plus clair
SAIN, DEGRADE, CRITIQUE = "#0ca30c", "#ec835a", "#d03b3b"
BLEU_TRANSPARENT = "rgba(57, 135, 229, 0.22)"                 # zones d'incertitude (intervalles)

# une nuance de bleu par page : dégradé du bandeau (texte blanc lisible partout)
# + une version claire pour les chiffres et les soulignements sur fond noir
COULEURS_PAGES = {
    "accueil": ("#0B1F4D", "#2563EB", "#6EA8FF"),       # bleu roi
    "donnees": ("#072A44", "#0369A1", "#4FB3F0"),       # bleu océan
    "sante": ("#06303D", "#0E7490", "#3CC6D9"),         # bleu pétrole
    "pronostic": ("#0F1E5A", "#1D4ED8", "#7A9BFF"),     # bleu cobalt
    "comparaison": ("#13205E", "#3949C7", "#8E9BFF"),   # bleu indigo
    "economie": ("#0A2547", "#1F6FD1", "#5EA8F7"),      # bleu azur
    "tests": ("#0E1E3D", "#2B5CA8", "#86A8E6"),         # bleu acier
}

STYLE = """
<style>
/* fond : noir avec un léger halo bleu en haut à droite */
[data-testid="stApp"] {background: radial-gradient(1100px 520px at 88% -8%, rgba(37, 99, 235, 0.13), transparent 62%), #0A0D13;}
[data-testid="stHeader"] {background: transparent;}
/* bandeau en haut de chaque page */
.bandeau {border-radius: 18px; padding: 22px 28px 20px; margin: 0 0 6px 0; color: #fff;
          border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 10px 30px rgba(29, 78, 216, 0.22);}
.bandeau h1 {color: #fff !important; margin: 4px 0 0 0; padding: 0; font-size: 2.05rem; line-height: 1.2;
             font-family: 'Source Serif 4', 'Source Serif Pro', Georgia, serif; font-weight: 700;}
.bandeau p {color: rgba(255, 255, 255, 0.9); margin: 8px 0 0 0; font-size: 1rem; max-width: 64rem;}
.etiquette {display: inline-block; background: rgba(255, 255, 255, 0.14); border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 2px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
            letter-spacing: 0.05em; text-transform: uppercase;}
/* cartes des indicateurs (st.metric) */
[data-testid="stMetric"] {background: #111826; border: 1px solid #1F2A3B; border-left: 6px solid var(--accent);
                          border-radius: 14px; padding: 12px 16px;}
[data-testid="stMetricLabel"] p {color: #9AA6B8; font-weight: 600;}
[data-testid="stMetricValue"] {color: var(--accent);}
/* titres de section soulignés de la couleur de la page */
[data-testid="stHeading"] h3 {border-bottom: 3px solid var(--accent); display: inline-block; padding-bottom: 2px;}
/* graphiques posés sur des cartes sombres */
[data-testid="stPlotlyChart"] {background: #10151F; border: 1px solid #1F2A3B; border-radius: 14px; padding: 6px 8px;}
/* cartes en HTML (accueil) */
.carte {background: #111826; border: 1px solid #1F2A3B; border-top: 6px solid var(--couleur); border-radius: 14px;
        padding: 14px 16px 12px; min-height: 150px;}
.carte .numero {display: inline-block; background: var(--couleur); color: #fff; border-radius: 999px;
                padding: 1px 10px; font-size: 0.78rem; font-weight: 700;}
.carte h4 {margin: 8px 0 4px 0; padding: 0; color: #E6EBF3; font-size: 1.08rem;}
.carte p {margin: 0; color: #AEB8C8; font-size: 0.92rem; line-height: 1.45;}
.signature {font-size: 0.82rem; color: #8A95A8; border-top: 1px dashed #263043; padding-top: 10px; margin-top: 6px;}
</style>
"""


def entete(titre, sous_titre, page, etiquette=""):
    """Bandeau coloré en haut de page (nuance de bleu propre à chaque page) + style commun."""
    fonce, clair, vif = COULEURS_PAGES[page]
    st.html(STYLE + f"<style>:root {{--accent: {vif};}}</style>")
    st.html(f"""
    <div class="bandeau" style="background: linear-gradient(120deg, {fonce} 0%, {clair} 100%);">
      <span class="etiquette">{etiquette}</span>
      <h1>{titre}</h1>
      <p>{sous_titre}</p>
    </div>""")


def carte(numero, titre, texte, couleur):
    """Petite carte sombre avec un liseré de couleur en haut (page d'accueil)."""
    return f"""<div class="carte" style="--couleur: {couleur};">
      <span class="numero">{numero}</span><h4>{titre}</h4><p>{texte}</p></div>"""


# ---------------------------------------------------------------- chargement (mis en cache par Streamlit)
# Le cache dépend aussi de la date de modification du fichier : si on relance le pipeline
# pendant que le dashboard est ouvert, il relit tout seul les nouveaux résultats.
@st.cache_data
def lire_fichier(chemin, date_modification):
    if chemin.suffix == ".csv":
        return pd.read_csv(chemin)
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def lire(chemin):
    return lire_fichier(chemin, chemin.stat().st_mtime)


def charger(nom):
    """Lit un fichier JSON du dossier results/."""
    return lire(RESULTATS / nom)


def charger_series():
    return lire(RESULTATS / "series.csv")


def charger_signaux():
    return lire(TRAITE / "signaux.json")


def roulements():
    return pd.DataFrame(charger("donnees.json")["roulements"])


# ---------------------------------------------------------------- style des graphiques
def styliser(fig, hauteur=380, titre_x=None, titre_y=None, legende=True):
    """Même style pour tous les graphiques : fond sombre, grille discrète, légende en haut."""
    fig.update_layout(
        height=hauteur, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=FOND, plot_bgcolor=FOND,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=ENCRE),
        showlegend=legende, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title_text="",
                    traceorder="normal"),   # sinon Plotly inverse la légende dès qu'une zone est remplie
        hoverlabel=dict(bgcolor="#1A2232", bordercolor=AXE, font=dict(size=12, color=ENCRE)),
    )
    fig.update_xaxes(title_text=titre_x, gridcolor=GRILLE, linecolor=AXE, zeroline=False,
                     tickfont=dict(color=ENCRE_2))
    fig.update_yaxes(title_text=titre_y, gridcolor=GRILLE, linecolor=AXE, zeroline=False,
                     tickfont=dict(color=ENCRE_2))
    return fig


def ligne(x, y, nom, couleur, epaisseur=2, pointilles=False, **options):
    """Une courbe Plotly avec les réglages habituels."""
    return go.Scatter(x=x, y=y, name=nom, mode="lines",
                      line=dict(color=couleur, width=epaisseur, dash="dash" if pointilles else "solid"),
                      **options)


def pastille(etat):
    """Petit badge texte + icône pour l'état d'un roulement (jamais la couleur seule)."""
    if etat == "sain":
        return ":green[● sain]"
    return ":orange[▲ dégradé]"


def transparent(couleur, opacite):
    """'#3987e5' -> 'rgba(57, 135, 229, 0.35)' (même couleur, plus ou moins transparente)."""
    r, v, b = (int(couleur[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {v}, {b}, {opacite})"

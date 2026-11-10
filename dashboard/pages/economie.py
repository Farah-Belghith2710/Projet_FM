"""Sprint 4 — Décision de maintenance : quelle stratégie coûte le moins cher ?"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from commun import COULEURS_MODELES, COULEURS_PAGES, ENCRE, ENCRE_2, FOND, charger, entete, roulements, styliser
from src import economie

entete("Décision de maintenance et économie",
       "On rejoue la vie des 11 roulements de test avec chaque stratégie : remplacer trop tard coûte une panne, "
       "remplacer trop tôt gaspille de la durée de vie. Tous les coûts sont en unités relatives (1 u = un "
       "remplacement planifié).", "economie", "Sprint 4 · Décision")

rul, sante, fiab = charger("rul.json"), charger("sante.json"), charger("fiabilite.json")
tab = roulements().set_index("roulement")
beta, eta = fiab["weibull"]["beta"], fiab["weibull"]["eta_min"]
noms = list(rul["roulements"])
vies = [float(tab.loc[r, "duree_vie_s"]) / 60 for r in noms]
fpt = [sante["roulements"][r]["fpt_index"] * 10 / 60 for r in noms]          # un fichier toutes les 10 s
fausses = [len(sante["roulements"][r]["episodes"]) - 1 for r in noms]
courbes = [rul["courbes"][r] for r in noms]
modeles = [m for m in rul["modeles"] if m != "Fiabilité (B10)"]

# ---------------------------------------------------------------- hypothèses (modifiables)
st.subheader("Les hypothèses (modifiables)")
c1, c2, c3, c4 = st.columns(4)
cc = c1.slider("Coût d'une panne Cc (u)", 1.5, 10.0, 5.0, 0.5,
               help="Remplacement en urgence + arrêt de production + dégâts. Cp = 1 u.")
ci = c2.slider("Coût d'une fausse alarme (u)", 0.0, 0.5, 0.1, 0.05,
               help="Une inspection pour rien quand l'alarme est annulée")
delai = c3.slider("Délai d'intervention (min)", 0.0, 10.0, 2.0, 0.5,
                  help="Temps entre la décision et le remplacement effectif (échelle de l'essai accéléré)")
seuil = c4.slider("Seuil de décision s (min)", 2.0, 60.0, 10.0, 1.0,
                  help="On déclenche le remplacement quand la RUL prédite passe sous ce seuil")

resultats, t_etoile = economie.comparer_strategies(vies, fpt, courbes, fausses, beta, eta, cp=1.0, cc=cc, ci=ci,
                                                   delai_min=delai, seuil_min=seuil, modeles=modeles)
tableau = pd.DataFrame(resultats).T.reset_index(names="strategie")
realistes = tableau[~tableau["strategie"].str.startswith("Idéale")]
meilleure = realistes.sort_values("cout_par_heure_u").iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Âge optimal T* (préventif)", f"{t_etoile:.0f} min", help="Minimise le coût par heure avec la loi de Weibull")
nom = meilleure["strategie"]                       # nom court pour que la carte reste lisible
court = nom.replace("Prédictive (", "").replace(" (âge T*)", "").replace(" (dès le FPT)", "").rstrip(")")
m2.metric("Meilleure stratégie réaliste", court, help=f"{nom} : la moins chère des stratégies applicables en pratique")
m3.metric("Gain par rapport au correctif", f"{meilleure['gain_vs_corrective_pct']:.0f} %")
m4.metric("Pannes évitées", f"{meilleure['pannes_evitees_pct']:.0f} %",
          help="Part des 11 roulements remplacés avant leur panne")

# ---------------------------------------------------------------- comparaison des stratégies
GRIS = {"Corrective": "#A7B0BF", "Préventive (âge T*)": "#8B95A6", "Conditionnelle (dès le FPT)": "#737E90",
        "Idéale (juste avant la panne)": "#5E697B", "Prédictive (Wiener prudent, 5 %)": "#86b6ef"}


def couleur(strategie):
    if strategie in GRIS:
        return GRIS[strategie]
    return COULEURS_MODELES[strategie.replace("Prédictive (", "").rstrip(")")]


gauche, droite = st.columns([1.25, 1])
with gauche:
    st.markdown("**Coût par heure de fonctionnement de chaque stratégie** (plus court = mieux)")
    t = tableau.sort_values("cout_par_heure_u", ascending=False)
    fig = go.Figure(go.Bar(
        y=t["strategie"], x=t["cout_par_heure_u"], orientation="h", marker_cornerradius=4,
        marker_color=[couleur(s) for s in t["strategie"]],
        text=[f"{c:.2f} u/h · {p:.0f} panne(s)" for c, p in zip(t["cout_par_heure_u"], t["pannes"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y} : %{x:.3f} u/h<extra></extra>"))
    styliser(fig, hauteur=440, titre_x="coût par heure (u/h)", legende=False)
    fig.update_layout(hovermode="closest", margin=dict(l=10, r=40, t=20, b=10))
    fig.update_xaxes(range=[0, float(t["cout_par_heure_u"].max()) * 1.75])
    st.plotly_chart(fig)
with droite:
    st.markdown("**Remplacement préventif à âge fixe : le coût dépend de l'âge choisi**")
    ages, couts = economie.couts_age(beta, eta, 1.0, cc, t_max=700)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages, y=couts, mode="lines", line=dict(color=COULEURS_PAGES["economie"][2], width=2.5),
                             name="coût par heure C(T)", hovertemplate="T = %{x:.0f} min : %{y:.3f} u/h<extra></extra>"))
    c_corrective = cc / fiab["weibull"]["mttf_min"] * 60          # tout correctif : Cc / MTTF
    fig.add_hline(y=c_corrective, line=dict(color=ENCRE_2, width=1, dash="dot"))
    fig.add_annotation(x=0.96, xref="x domain", y=c_corrective, text="tout correctif", showarrow=False,
                       xanchor="right", yanchor="bottom", font_color=ENCRE_2)
    fig.add_trace(go.Scatter(x=[t_etoile], y=[float(np.min(couts))], mode="markers+text", name="T* optimal",
                             marker=dict(size=12, color=ENCRE, line=dict(color=FOND, width=2)),
                             text=[f"T* = {t_etoile:.0f} min"], textposition="top center"))
    styliser(fig, hauteur=440, titre_x="âge de remplacement T (min)", titre_y="coût par heure (u/h)")
    fig.update_yaxes(range=[0, c_corrective * 2.2])
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig)

st.dataframe(
    tableau.sort_values("cout_par_heure_u"), hide_index=True,
    column_config={"strategie": "Stratégie",
                   "cout_total_u": st.column_config.NumberColumn("Coût total (u)", format="%.1f"),
                   "cout_par_heure_u": st.column_config.NumberColumn("Coût par heure (u/h)", format="%.3f"),
                   "pannes": "Pannes (sur 11)",
                   "pannes_evitees_pct": st.column_config.NumberColumn("Pannes évitées", format="%.0f %%"),
                   "vie_perdue_moyenne_min": st.column_config.NumberColumn("Vie gaspillée moyenne (min)",
                                                                           format="%.1f"),
                   "gain_vs_corrective_pct": st.column_config.NumberColumn("Gain vs correctif", format="%.0f %%")})
st.caption("Préventive : on remplace tous les roulements à l'âge T* (même s'ils sont encore bons). Conditionnelle : "
           "on remplace dès l'alarme du FPT. Prédictive : on remplace quand la RUL prédite passe sous le seuil s. "
           "« Wiener prudent » décide avec la borne basse (5 %) de l'intervalle de prédiction. Idéale : impossible en "
           "pratique, c'est la limite à viser.")

# ---------------------------------------------------------------- sensibilité au seuil
st.subheader("Effet du seuil de décision s")
seuils = [2, 5, 8, 10, 15, 20, 30, 45, 60]
lignes = []
for s in seuils:
    bilan, _ = economie.comparer_strategies(vies, fpt, courbes, fausses, beta, eta, cp=1.0, cc=cc, ci=ci,
                                            delai_min=delai, seuil_min=s, modeles=modeles)
    lignes.append({nom: b["cout_par_heure_u"] for nom, b in bilan.items()} | {"seuil": s})
sensibilite = pd.DataFrame(lignes)
fig = go.Figure()
for modele in modeles:
    fig.add_trace(go.Scatter(x=sensibilite["seuil"], y=sensibilite[f"Prédictive ({modele})"], mode="lines+markers",
                             name=modele, line=dict(color=COULEURS_MODELES[modele], width=2),
                             marker=dict(size=7, line=dict(color=FOND, width=1.5))))
fig.add_trace(go.Scatter(x=sensibilite["seuil"], y=sensibilite["Conditionnelle (dès le FPT)"], mode="lines",
                         name="conditionnelle (dès le FPT)", line=dict(color=ENCRE_2, width=1.5, dash="dash")))
styliser(fig, hauteur=360, titre_x="seuil de décision s (min)", titre_y="coût par heure (u/h)")
st.plotly_chart(fig)
st.caption("Un seuil trop petit déclenche trop tard : le roulement casse avant l'intervention. Un seuil trop grand "
           "revient à remplacer dès l'alarme. Le bon seuil dépend du modèle : il se règle comme un hyperparamètre.")

st.info("**À retenir.** Surveiller le roulement (conditionnel ou prédictif) divise le coût par deux par rapport au "
        "correctif. Les pannes restantes sont les roulements dont le défaut apparaît moins de 2 min avant la casse : "
        "aucun modèle ne laisse le temps d'intervenir. Les coûts sont des hypothèses d'illustration : à remplacer "
        "par les vrais coûts de l'usine.", icon=":material/savings:")

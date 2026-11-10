"""Sprint 3 — Prédictions de la RUL pour chaque roulement de test."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from commun import BLEU_TRANSPARENT, COULEURS_MODELES, ENCRE, FOND, charger, entete, pastille, styliser

entete("Pronostic de la durée de vie résiduelle",
       "Avant le FPT, le roulement paraît sain : seule la fiabilité de la population (B10 conditionnel) peut estimer sa RUL. "
       "Après le FPT, les modèles utilisent les mesures vibratoires.", "pronostic", "Sprint 3 · Modèles")

rul = charger("rul.json")
noms_modeles = rul["modeles"]
troncature = {ligne["roulement"]: ligne for ligne in rul["troncature"]}

gauche, droite = st.columns([1, 2])
with gauche:
    choix = st.selectbox("Roulement de test", rul["roulements"])
with droite:
    modeles = st.multiselect("Modèles affichés", noms_modeles, default=noms_modeles)
intervalle = st.toggle("Afficher l'intervalle de prédiction à 90 % du modèle de Wiener", value=False)

c = rul["courbes"][choix]
t = troncature[choix]
depuis = np.array(c["temps_min"]) - c["temps_min"][0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("FPT détecté", f"{c['fpt_min']:.1f} min")
c2.metric("Phase de dégradation", f"{max(c['vrai_min']):.1f} min")
c3.metric("Troncature du challenge", f"{c['troncature_min']:.1f} min")
c4.metric("RUL officielle à la troncature", f"{t['rul_officielle_min']:.1f} min")
st.markdown(f"État à l'instant de troncature : {pastille(t['etat'])}")

# ---------------------------------------------------------------- courbes de RUL
fig = go.Figure()
if intervalle and "Wiener" in modeles:
    fig.add_trace(go.Scatter(x=depuis, y=c["wiener_haut"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=depuis, y=c["wiener_bas"], line=dict(width=0), fill="tonexty",
                             fillcolor=BLEU_TRANSPARENT, name="Wiener : intervalle 90 %", hoverinfo="skip",
                             legendrank=1001))
for nom in modeles:
    fig.add_trace(go.Scatter(x=depuis, y=c["predictions"][nom], name=nom, mode="lines",
                             line=dict(color=COULEURS_MODELES[nom], width=2,
                                       dash="dash" if nom == "Fiabilité (B10)" else "solid"),
                             hovertemplate="%{y:.1f} min"))
fig.add_trace(go.Scatter(x=depuis, y=c["vrai_min"], name="RUL réelle", mode="lines",
                         line=dict(color=ENCRE, width=3), hovertemplate="%{y:.1f} min", legendrank=1))   # 1er dans la légende
points = c["points_evaluation"]
fig.add_trace(go.Scatter(x=depuis[points], y=np.array(c["vrai_min"])[points], mode="markers",
                         name="points d'évaluation (25, 50, 75 %)",
                         marker=dict(size=10, color=ENCRE, symbol="diamond", line=dict(color=FOND, width=2)),
                         hoverinfo="skip", legendrank=1002))
haut = max(c["vrai_min"]) * 2.5
styliser(fig, hauteur=460, titre_x="temps depuis le FPT (min)", titre_y="RUL (min)")
fig.update_yaxes(range=[0, haut])
st.plotly_chart(fig)
st.caption("L'axe vertical est limité à 2,5 fois la durée de la phase de dégradation ; survoler le graphique "
           "pour lire toutes les valeurs.")

# ---------------------------------------------------------------- protocole du challenge
st.subheader("Prédiction à l'instant de troncature (protocole officiel)")
if t["etat"] == "sain":
    st.warning("À la troncature, aucun défaut n'est encore détecté sur ce roulement : tous les modèles "
               "utilisent donc la fiabilité de la population (B10 conditionnel : durée que le roulement tiendra "
               "encore avec 90 % de chances).", icon=":material/info:")
lignes = []
for nom in noms_modeles:
    lignes.append({"Modèle": nom, "RUL prédite (min)": t[nom]["rul_predite_min"],
                   "RUL officielle (min)": t["rul_officielle_min"], "%Er": t[nom]["erreur_pct"],
                   "Score A": t[nom]["score"]})
st.dataframe(pd.DataFrame(lignes), hide_index=True,
             column_config={"RUL prédite (min)": st.column_config.NumberColumn(format="%.1f"),
                            "RUL officielle (min)": st.column_config.NumberColumn(format="%.1f"),
                            "%Er": st.column_config.NumberColumn(format="%.1f %%"),
                            "Score A": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
st.caption("%Er = 100 × (RUL réelle − RUL prédite) / RUL réelle. Négatif : prédiction trop tardive, "
           "lourdement pénalisée par le score du challenge.")

# ---------------------------------------------------------------- hyperparamètres
with st.expander("Hyperparamètres choisis par validation « un roulement de côté »"):
    infos = rul["infos_modeles"]
    lignes = []
    for nom in noms_modeles:
        if nom not in infos:
            continue
        i = infos[nom]
        if nom == "Wiener":
            par = i["parametres"]
            texte = (f"seuil D = {par['seuil']:.2f}, dérive a priori {par['mu0']:.3f} ± {par['s0']:.3f} /min, "
                     f"σ = {par['sigma']:.3f}")
        else:
            texte = ", ".join(f"{k} = {v}" for k, v in i["meilleurs_parametres"].items())
        lignes.append({"Modèle": nom, "Réglages": texte, "Combinaisons testées": i.get("n_combinaisons", 1),
                       "Durée d'apprentissage (s)": i["duree_s"]})
    st.dataframe(pd.DataFrame(lignes), hide_index=True)
    st.caption(f"Entrées des modèles ML et deep learning : {', '.join(rul['entrees'])}. "
               "Cible : log(1 + RUL en minutes). Les réseaux LSTM et CNN 1D voient en plus une fenêtre "
               "glissante des dernières mesures.")

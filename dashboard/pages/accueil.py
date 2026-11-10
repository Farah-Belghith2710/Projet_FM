"""Vue d'ensemble : le projet en une page."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from commun import (COULEURS_CONDITIONS, COULEURS_MODELES, COULEURS_PAGES, ENCRE, ENCRE_2, carte, charger, charger_series,
                    entete, roulements, styliser)

entete("Pronostic de la durée de vie résiduelle (RUL) de roulements",
       "Étude comparative sur les données FEMTO-ST / PRONOSTIA du challenge IEEE PHM 2012 : "
       "fiabilité, détection du premier défaut, modèles stochastique, ML classique et deep learning, "
       "puis décision de maintenance.", "accueil", "Vue d'ensemble")

tab = roulements()
rul = charger("rul.json")
metriques = pd.DataFrame(rul["metriques"])
meilleur = metriques.sort_values("score_phase", ascending=False).iloc[0]
plus_precis = metriques.sort_values("rmse_min").iloc[0]
troncature = pd.DataFrame(rul["troncature"])
n_sains = int((troncature["etat"] == "sain").sum())

# ---------------------------------------------------------------- chiffres clés
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Roulements", f"{len(tab)}", help="6 d'apprentissage (essais complets) et 11 de test")
c2.metric("Fichiers de vibration", f"{len(charger_series()):,}".replace(",", " "),
          help="Un fichier = 0,1 s de signal (2 560 points à 25,6 kHz), toutes les 10 s")
c3.metric("Durée de vie (min)", f"{tab['duree_vie_min'].min():.0f} – {tab['duree_vie_min'].max():.0f}",
          help="Très variable, même sous la même condition opératoire")
c4.metric("Meilleur modèle", meilleur["modele"], help="Selon le score PHM sur la phase de dégradation")
c5.metric("Son score PHM", f"{meilleur['score_phase']:.3f}",
          help="Score du challenge (1 = parfait) à 25, 50 et 75 % de la phase de dégradation")

st.info(f"**À retenir.** Au moment de la troncature du challenge, **{n_sains} roulements de test sur 11** "
        "ne montrent encore aucun défaut : aucune méthode fondée sur les capteurs ne peut les anticiper, "
        "seule la fiabilité de la population répond. Après le premier défaut (FPT), le meilleur modèle "
        f"selon le score PHM est **{meilleur['modele']}**, et le plus précis en RMSE est "
        f"**{plus_precis['modele']}** ({plus_precis['rmse_min']:.1f} min).", icon=":material/lightbulb:")

# ---------------------------------------------------------------- démarche
st.subheader("Démarche en 4 sprints, de A à Z")
cartes = [
    ("Sprint 1", "Données et fiabilité", "24 889 fichiers, contrôle qualité, filtre 10 Hz – 10 kHz, "
     "Kaplan-Meier, Weibull, B10, B50, L10.", "donnees", "pages/donnees.py"),
    ("Sprint 2", "Santé et premier défaut", "34 features, normalisation par l'état sain, FPT par la règle "
     "des 3 σ, indicateur de santé.", "sante", "pages/sante.py"),
    ("Sprint 3", "Modèles de RUL", "Wiener bayésien, SVR, GPR, Random Forest, LSTM et CNN 1D, réglés en "
     "laissant un roulement de côté.", "pronostic", "pages/pronostic.py"),
    ("Sprint 4", "Comparaison des modèles", "RMSE, MAE, score PHM asymétrique, bootstrap, explicabilité "
     "(XAI) et covariables.", "comparaison", "pages/comparaison.py"),
    ("Sprint 4", "Décision et économie", "Correctif, préventif à l'âge T*, conditionnel ou prédictif : quelle "
     "stratégie coûte le moins ?", "economie", "pages/economie.py"),
    ("Sprint 4", "Tests et validation", "Données simulées, tests statistiques et tests automatiques du code.",
     "tests", "pages/tests.py"),
]
for ligne_cartes in (cartes[:3], cartes[3:]):
    for colonne, (numero, titre, texte, page, fichier) in zip(st.columns(3), ligne_cartes):
        with colonne:
            st.html(carte(numero, titre, texte, COULEURS_PAGES[page][1]))
            st.page_link(fichier, label="Ouvrir la page", icon=":material/arrow_forward:")

# ---------------------------------------------------------------- durées de vie
gauche, droite = st.columns([1.15, 1])
with gauche:
    st.subheader("Durées de vie des 17 roulements")
    t = tab.sort_values(["condition", "duree_vie_min"], ascending=[False, True])
    fig = go.Figure()
    for condition, groupe in t.groupby("condition"):
        fig.add_trace(go.Bar(y=groupe["roulement"], x=groupe["duree_vie_min"], orientation="h",
                             name=f"condition {condition}", marker_color=COULEURS_CONDITIONS[condition],
                             marker_cornerradius=4,
                             hovertemplate="%{y} : %{x:.0f} min<extra></extra>"))
    test = t[t["ensemble"] == "test"]
    fig.add_trace(go.Scatter(y=test["roulement"], x=test["troncature"] * 10 / 60, mode="markers",
                             name="troncature (challenge)", marker=dict(symbol="line-ns", size=14, color=ENCRE,
                                                                        line=dict(width=3, color=ENCRE)),
                             hovertemplate="troncature à %{x:.0f} min<extra></extra>"))
    styliser(fig, hauteur=520, titre_x="durée de vie (min)")
    fig.update_layout(hovermode="closest", barmode="overlay", bargap=0.35)
    fig.update_yaxes(categoryorder="array", categoryarray=t["roulement"].tolist())
    st.plotly_chart(fig)

with droite:
    st.subheader("Classement des modèles")
    t = metriques.sort_values("score_phase")
    fig = go.Figure(go.Bar(
        y=t["modele"], x=t["score_phase"], orientation="h", marker_cornerradius=4,
        marker_color=[COULEURS_MODELES[m] for m in t["modele"]],
        error_x=dict(type="data", symmetric=False, color=ENCRE_2, thickness=1,
                     array=[ic[1] - v for ic, v in zip(t["score_phase_ic95"], t["score_phase"])],
                     arrayminus=[v - ic[0] for ic, v in zip(t["score_phase_ic95"], t["score_phase"])]),
        hovertemplate="%{y} : %{x:.3f}<extra></extra>"))
    styliser(fig, hauteur=330, titre_x="score PHM (1 = parfait)", legende=False)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig)
    st.caption("Barres d'erreur : intervalle de confiance à 95 % obtenu par bootstrap sur les 11 roulements de test.")
    st.dataframe(
        metriques[["modele", "rmse_min", "mae_min", "score_phase", "score_challenge"]]
        .sort_values("score_phase", ascending=False),
        hide_index=True,
        column_config={"modele": "Modèle", "rmse_min": st.column_config.NumberColumn("RMSE (min)", format="%.1f"),
                       "mae_min": st.column_config.NumberColumn("MAE (min)", format="%.1f"),
                       "score_phase": st.column_config.NumberColumn("Score PHM", format="%.3f"),
                       "score_challenge": st.column_config.NumberColumn("Score challenge", format="%.3f")})

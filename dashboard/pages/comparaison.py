"""Sprint 4 — Comparaison des modèles : métriques PHM, incertitude, coût de calcul."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from commun import COULEURS_MODELES, DEGRADE, ENCRE, ENCRE_2, FOND, SAIN, charger, entete, styliser

entete("Comparaison des modèles",
       "Deux protocoles sur les 11 roulements de test : (1) toute la phase de dégradation, à partir du FPT ; "
       "(2) le protocole officiel du challenge, une seule prédiction à l'instant de troncature.",
       "comparaison", "Sprint 4 · Évaluation")

rul = charger("rul.json")
metriques = pd.DataFrame(rul["metriques"])
noms = rul["modeles"]


def barres(colonne, intervalle, titre, croissant):
    """Barres horizontales triées, avec l'intervalle de confiance bootstrap si disponible."""
    t = metriques.sort_values(colonne, ascending=not croissant)
    options = {}
    if intervalle:
        options["error_x"] = dict(type="data", symmetric=False, color=ENCRE_2, thickness=1,
                                  array=[ic[1] - v for ic, v in zip(t[intervalle], t[colonne])],
                                  arrayminus=[v - ic[0] for ic, v in zip(t[intervalle], t[colonne])])
    fig = go.Figure(go.Bar(y=t["modele"], x=t[colonne], orientation="h", marker_cornerradius=4,
                           marker_color=[COULEURS_MODELES[m] for m in t["modele"]],
                           hovertemplate="%{y} : %{x:.3f}<extra></extra>", **options))
    styliser(fig, hauteur=320, titre_x=titre, legende=False)
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------- protocole 1
st.subheader("Protocole 1 · Phase de dégradation (du FPT à la défaillance)")
st.markdown("RMSE et MAE : moyennes sur les roulements de l'erreur en minutes. Score PHM : moyenne du score "
            "du challenge calculé à 25 %, 50 % et 75 % de la phase de dégradation de chaque roulement.")
c1, c2 = st.columns(2)
c1.plotly_chart(barres("score_phase", "score_phase_ic95", "score PHM (plus haut = mieux)", True))
c2.plotly_chart(barres("rmse_min", "rmse_ic95", "RMSE en minutes (plus bas = mieux)", False))
st.caption("Barres d'erreur : intervalles de confiance à 95 % par bootstrap (on retire au hasard, avec remise, "
           "les 11 roulements de test). Des intervalles qui se chevauchent veulent dire que l'écart entre deux "
           "modèles n'est pas significatif avec si peu de roulements.")

st.markdown("**RMSE par roulement (minutes)**")
valeurs = np.array([rul["par_roulement"][n]["rmse_min"] for n in noms])
fig = go.Figure(go.Heatmap(z=np.log10(valeurs + 1), x=rul["roulements"], y=noms, text=valeurs,
                           texttemplate="%{text:.1f}", textfont=dict(color="#FFFFFF"), showscale=False,
                           colorscale=[[0, "#121C2E"], [1, "#2F6FD0"]],   # plus l'erreur est grande, plus c'est vif
                           hovertemplate="%{y} — %{x} : %{text:.1f} min<extra></extra>"))
styliser(fig, hauteur=330, legende=False)
fig.update_layout(hovermode="closest")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False, autorange="reversed")
st.plotly_chart(fig)
st.caption("Couleur en échelle logarithmique. Bearing1_3 (près de 3 h de dégradation lente) concentre "
           "les plus grandes erreurs ; les défaillances brutales (quelques minutes) sont plus faciles en minutes, "
           "mais plus dures en pourcentage.")

couverture = rul["couverture_wiener_90"]["moyenne"]
st.metric("Couverture réelle de l'intervalle à 90 % du modèle de Wiener", f"{100 * couverture:.0f} %",
          help="Part des instants où la RUL réelle tombe dans l'intervalle prédit (idéalement 90 %)")

# ---------------------------------------------------------------- protocole 2
st.subheader("Protocole 2 · Challenge IEEE PHM 2012 (instant de troncature)")
tronc = rul["troncature"]
n_sains = sum(1 for ligne in tronc if ligne["etat"] == "sain")
st.markdown(f"À la troncature, **{n_sains} roulements sur 11 sont encore sains** (aucun défaut détecté) : "
            "pour eux, tous les modèles donnent la même prédiction (fiabilité). Les écarts entre modèles "
            f"ne viennent donc que des {11 - n_sains} roulements déjà dégradés.")
c1, c2 = st.columns([1, 1.6])
c1.plotly_chart(barres("score_challenge", "score_challenge_ic95", "score du challenge (1 = parfait)", True))
with c2:
    fig = go.Figure()
    x = np.arange(len(tronc))
    roulements_tronc = [ligne["roulement"] for ligne in tronc]
    for decalage, nom in zip(np.linspace(-0.3, 0.3, len(noms)), noms):   # petit décalage : aucun point caché
        erreurs = np.clip([ligne[nom]["erreur_pct"] for ligne in tronc], -300, 100)
        fig.add_trace(go.Scatter(x=x + decalage, y=erreurs, mode="markers", name=nom, text=roulements_tronc,
                                 marker=dict(size=8, color=COULEURS_MODELES[nom], line=dict(color=FOND, width=1)),
                                 hovertemplate=nom + " — %{text} : %{y:.0f} %<extra></extra>"))
    for i, ligne in enumerate(tronc):
        fig.add_vrect(x0=i - 0.45, x1=i + 0.45, fillcolor=SAIN if ligne["etat"] == "sain" else DEGRADE,
                      opacity=0.12, line_width=0)
    fig.add_hline(y=0, line=dict(color=ENCRE, width=1))
    styliser(fig, hauteur=340, titre_y="%Er (bornée à −300 %)")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickvals=x, ticktext=roulements_tronc, showgrid=False)
    st.plotly_chart(fig)
st.caption("Fond vert : roulement sain à la troncature ; fond orange : roulement déjà dégradé. "
           "%Er < 0 : prédiction trop tardive (dangereuse).")
reference = rul["reference_mrl"]
st.markdown(f"**Pourquoi le B10 conditionnel et pas la moyenne ?** Si les roulements sains utilisaient la durée de "
            f"vie résiduelle moyenne (MRL), le score du challenge de la fiabilité seule tomberait à "
            f"**{reference['score_challenge']:.3f}** (|%Er| moyen {reference['erreur_abs_moyenne_pct']:.0f} %) : "
            "la moyenne prédit presque toujours trop tard, ce que le score punit très fort.")

gauche, droite = st.columns(2)
with gauche:
    st.markdown("**Fonction de score du challenge**")
    er = np.linspace(-50, 50, 300)
    a = np.where(er <= 0, np.exp(-np.log(0.5) * er / 5), np.exp(np.log(0.5) * er / 20))
    fig = go.Figure(go.Scatter(x=er, y=a, mode="lines", line=dict(color=COULEURS_MODELES["Wiener"], width=2),
                               hovertemplate="%Er = %{x:.0f} % → A = %{y:.2f}<extra></extra>"))
    fig.add_annotation(x=-48, y=0.9, xanchor="left", text="tardive : pénalité forte", showarrow=False, font_color=ENCRE_2)
    fig.add_annotation(x=48, y=0.9, xanchor="right", text="précoce : pénalité douce", showarrow=False, font_color=ENCRE_2)
    styliser(fig, hauteur=280, titre_x="%Er", titre_y="score A", legende=False)
    st.plotly_chart(fig)
with droite:
    st.markdown("**Coût de calcul (apprentissage, recherche d'hyperparamètres comprise)**")
    infos = rul["infos_modeles"]
    t = pd.DataFrame([{"modele": n, "duree": infos[n]["duree_s"]} for n in noms if n in infos])
    t = t.sort_values("duree")
    fig = go.Figure(go.Bar(y=t["modele"], x=t["duree"], orientation="h", marker_cornerradius=4,
                           marker_color=[COULEURS_MODELES[m] for m in t["modele"]],
                           hovertemplate="%{y} : %{x:.1f} s<extra></extra>"))
    styliser(fig, hauteur=280, titre_x="secondes (échelle log)", legende=False)
    fig.update_xaxes(type="log")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig)

# ---------------------------------------------------------------- tableau complet
st.subheader("Tableau récapitulatif")
tableau = metriques.copy()
tableau["IC 95 % score PHM"] = [f"[{a:.3f} ; {b:.3f}]" for a, b in tableau["score_phase_ic95"]]
tableau["IC 95 % challenge"] = [f"[{a:.3f} ; {b:.3f}]" for a, b in tableau["score_challenge_ic95"]]
st.dataframe(
    tableau[["modele", "famille", "rmse_min", "mae_min", "score_phase", "IC 95 % score PHM",
             "score_challenge", "IC 95 % challenge", "erreur_abs_moyenne_pct"]],
    hide_index=True,
    column_config={"modele": "Modèle", "famille": "Famille",
                   "rmse_min": st.column_config.NumberColumn("RMSE (min)", format="%.1f"),
                   "mae_min": st.column_config.NumberColumn("MAE (min)", format="%.1f"),
                   "score_phase": st.column_config.NumberColumn("Score PHM", format="%.3f"),
                   "score_challenge": st.column_config.NumberColumn("Score challenge", format="%.3f"),
                   "erreur_abs_moyenne_pct": st.column_config.NumberColumn(
                       "|%Er| moyen", help="Erreur absolue moyenne en %, protocole du challenge", format="%.0f %%")})

# ---------------------------------------------------------------- explicabilité (XAI)
comp = charger("complements.json")
st.subheader("Explicabilité (XAI) : sur quoi les modèles s'appuient-ils ?")
xai = comp["explicabilite"]
gauche, droite = st.columns([1.3, 1])
with gauche:
    choix = st.radio("Modèle", ["SVR", "GPR", "Random Forest"], horizontal=True, key="xai")
    imp = pd.DataFrame(xai[choix]["importances"]).iloc[::-1]
    fig = go.Figure(go.Bar(y=imp["nom"], x=imp["hausse_mae_min"], orientation="h", marker_cornerradius=4,
                           marker_color=COULEURS_MODELES[choix],
                           error_x=dict(type="data", array=imp["ecart_type"], color=ENCRE_2, thickness=1),
                           hovertemplate="%{y} : +%{x:.2f} min d'erreur<extra></extra>"))
    styliser(fig, hauteur=360, titre_x="hausse de la MAE (min) si on mélange l'entrée", legende=False)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig)
with droite:
    st.markdown("**Importance par permutation** : on mélange au hasard une entrée (on casse son lien avec la RUL) et "
                "on regarde de combien l'erreur augmente sur les roulements de test. Plus la barre est longue, plus "
                "le modèle a besoin de cette entrée.")
    st.markdown("- Les trois modèles s'appuient surtout sur des **énergies de bandes de fréquence** (ondelettes), "
                "pas seulement sur le temps écoulé depuis le FPT : ils lisent vraiment l'état du roulement.\n"
                "- Les bandes de 1,6 à 9,6 kHz dominent : c'est là que résonnent les chocs d'un défaut de roulement.\n"
                "- Le processus de Wiener est explicable par construction : une dérive, un seuil, un bruit.")
    st.caption(f"Calculé sur {xai['n_points_test']} instants des 11 roulements de test (10 mélanges par entrée).")

# ---------------------------------------------------------------- covariables dans les modèles ML
st.subheader("Covariables : ajouter la charge et la vitesse aux modèles ML")
cov = comp["covariables_ml"]
lignes = []
for modele in ["SVR", "GPR", "Random Forest"]:
    a, b = cov["sans covariables"][modele], cov["avec charge et vitesse"][modele]
    lignes.append({"modele": modele, "rmse_sans": a["rmse_min"], "rmse_avec": b["rmse_min"],
                   "score_sans": a["score_phase"], "score_avec": b["score_phase"],
                   "challenge_sans": a["score_challenge"], "challenge_avec": b["score_challenge"]})
st.dataframe(pd.DataFrame(lignes), hide_index=True, column_config={
    "modele": "Modèle", "rmse_sans": st.column_config.NumberColumn("RMSE sans (min)", format="%.2f"),
    "rmse_avec": st.column_config.NumberColumn("RMSE avec (min)", format="%.2f"),
    "score_sans": st.column_config.NumberColumn("Score PHM sans", format="%.3f"),
    "score_avec": st.column_config.NumberColumn("Score PHM avec", format="%.3f"),
    "challenge_sans": st.column_config.NumberColumn("Challenge sans", format="%.3f"),
    "challenge_avec": st.column_config.NumberColumn("Challenge avec", format="%.3f")})
st.caption("Le GPR progresse sur tous les critères ; le SVR gagne en score PHM mais perd en RMSE ; la forêt aléatoire "
           "change peu. Avec seulement 2 roulements d'apprentissage par condition, l'effet des covariables reste "
           "fragile (recalcul de l'étape 4 : légers écarts possibles avec le tableau principal pour la forêt).")

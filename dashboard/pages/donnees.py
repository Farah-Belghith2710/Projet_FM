"""Sprint 1 — Données brutes, prétraitement et fiabilité de la population."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from commun import (BLEU_TRANSPARENT, CATEGORIES, COULEURS_CONDITIONS, DISCRET, ENCRE, FOND, NEUTRE, charger, charger_series,
                    charger_signaux, entete, ligne, roulements, styliser)

entete("Données vibratoires et fiabilité",
       "Plateforme PRONOSTIA : 2 accéléromètres (horizontal et vertical), 25,6 kHz, 0,1 s enregistrée toutes les 10 s. "
       "Essais arrêtés quand la vibration dépasse 20 g.", "donnees", "Sprint 1 · Données")

tab = roulements()
series = charger_series()
signaux = charger_signaux()

# ---------------------------------------------------------------- tableau des roulements
st.subheader("Les 17 roulements")
affichage = tab[["roulement", "ensemble", "condition", "vitesse_rpm", "charge_N", "n_fichiers", "duree_vie_min",
                 "troncature", "rul_officielle_s", "fpt_min", "phase_degradation_min",
                 "premier_depassement_20g_s", "horodatages_anormaux"]].copy()
for colonne in ["troncature", "rul_officielle_s", "premier_depassement_20g_s"]:
    affichage[colonne] = ["—" if pd.isna(v) else f"{v:.0f}" for v in affichage[colonne]]   # case vide -> tiret
st.dataframe(
    affichage, hide_index=True,
    column_config={
        "roulement": "Roulement", "ensemble": "Ensemble", "condition": "Condition",
        "vitesse_rpm": "Vitesse (tr/min)", "charge_N": "Charge (N)", "n_fichiers": "Fichiers",
        "duree_vie_min": st.column_config.NumberColumn("Durée de vie (min)", format="%.1f"),
        "troncature": "Troncature (fichier)", "rul_officielle_s": "RUL officielle (s)",
        "fpt_min": st.column_config.NumberColumn("FPT (min)", format="%.1f"),
        "phase_degradation_min": st.column_config.NumberColumn("Dégradation (min)", format="%.1f"),
        "premier_depassement_20g_s": "1er dépassement de 20 g (s)",
        "horodatages_anormaux": "Horodatages anormaux",
    })
st.caption("Contrôle qualité : aucun fichier incomplet ni valeur manquante. Bearing1_1 contient 3 horodatages "
           "incohérents (le temps est donc reconstruit à partir du numéro de fichier, 10 s par fichier). "
           "Pour Bearing1_4, la RUL officielle (339 s) ne correspond pas au nombre de fichiers (2 890 s) : "
           "anomalie connue du jeu de données, on garde la valeur officielle pour le score du challenge.")

# ---------------------------------------------------------------- un roulement en détail

st.subheader("Explorer un roulement")
choix = st.selectbox("Roulement", tab["roulement"], index=0)
s = series[series["roulement"] == choix]

gauche, droite = st.columns(2)
with gauche:
    st.markdown("**Amplitude brute sur toute la vie** (traits gris : ±20 g)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["temps_min"], y=s["h_max_brut"], line=dict(color=CATEGORIES[0], width=0.6),
                             name="horizontal", fill=None, legendgroup="h"))
    fig.add_trace(go.Scatter(x=s["temps_min"], y=s["h_min_brut"], line=dict(color=CATEGORIES[0], width=0.6),
                             fill="tonexty", name="horizontal", legendgroup="h", showlegend=False))
    fig.add_trace(go.Scatter(x=s["temps_min"], y=s["v_max_brut"], line=dict(color=CATEGORIES[1], width=0.6),
                             name="vertical", legendgroup="v"))
    fig.add_trace(go.Scatter(x=s["temps_min"], y=s["v_min_brut"], line=dict(color=CATEGORIES[1], width=0.6),
                             fill="tonexty", name="vertical", legendgroup="v", showlegend=False))
    for niveau in (20, -20):
        fig.add_hline(y=niveau, line=dict(color=DISCRET, width=1))
    styliser(fig, hauteur=330, titre_x="temps (min)", titre_y="accélération (g)")
    st.plotly_chart(fig)

with droite:
    instant = st.radio("Instant du signal", ["début de vie", "milieu", "fin de vie"], horizontal=True)
    k = ["début de vie", "milieu", "fin de vie"].index(instant)
    signal = signaux[choix]
    st.markdown(f"**Signal de 0,1 s à t = {signal['instants_s'][k] / 60:.0f} min** : brut et filtré")
    t_ms = np.arange(len(signal["brut_h"][k])) / 25.6
    fig = go.Figure()
    fig.add_trace(ligne(t_ms, signal["brut_h"][k], "brut", NEUTRE, 1))
    fig.add_trace(ligne(t_ms, signal["filtre_h"][k], "filtré 10 Hz – 10 kHz", CATEGORIES[0], 1))
    styliser(fig, hauteur=280, titre_x="temps (ms)", titre_y="accélération horizontale (g)")
    st.plotly_chart(fig)

st.markdown("**Évolution des indicateurs classiques** (cliquer sur la légende pour ajouter le kurtosis)")
fig = go.Figure()
fig.add_trace(ligne(s["temps_min"], s["h_rms_brut"], "RMS horizontal", CATEGORIES[0], 1))
fig.add_trace(ligne(s["temps_min"], s["v_rms_brut"], "RMS vertical", CATEGORIES[1], 1))
fig.add_trace(ligne(s["temps_min"], s["h_kurtosis_brut"], "kurtosis horizontal", CATEGORIES[2], 1, visible="legendonly"))
styliser(fig, hauteur=300, titre_x="temps (min)", titre_y="valeur (échelle log)")
fig.update_yaxes(type="log", tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
st.plotly_chart(fig)

# ---------------------------------------------------------------- fiabilité
st.subheader("Fiabilité de la population")
fiab = charger("fiabilite.json")
w = fiab["weibull"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forme β (Weibull)", f"{w['beta']:.2f}", help="β > 1 : taux de défaillance croissant (usure)")
c2.metric("Échelle η", f"{w['eta_min']:.0f} min")
c3.metric("MTTF", f"{w['mttf_min']:.0f} min", help="Durée moyenne de fonctionnement avant défaillance")
c4.metric("B10", f"{w['b10_min']:.0f} min", help="Âge auquel 10 % des roulements ont défailli")
st.caption("Ajustement par maximum de vraisemblance avec censure : les 6 roulements d'apprentissage sont des "
           "défaillances, les 11 roulements de test sont censurés à la troncature (ils fonctionnaient encore).")

gauche, droite = st.columns(2)
with gauche:
    km, c = fiab["kaplan_meier"], fiab["courbes"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km["t"], y=km["haut"], mode="lines", line=dict(width=0, shape="hv"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=km["t"], y=km["bas"], mode="lines", line=dict(width=0, shape="hv"), fill="tonexty",
                             fillcolor=BLEU_TRANSPARENT, name="IC 95 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=km["t"], y=km["survie"], mode="lines",
                             line=dict(color=CATEGORIES[0], width=2, shape="hv"), name="Kaplan-Meier"))
    fig.add_trace(ligne(c["t_min"], c["fiabilite"], f"Weibull (β = {w['beta']:.2f})", ENCRE, 1.5))
    styliser(fig, hauteur=330, titre_x="temps (min)", titre_y="fiabilité R(t)")
    st.plotly_chart(fig)
with droite:
    choix_courbe = st.radio("Courbe", ["Taux de défaillance λ(t)", "Durée de vie résiduelle"], horizontal=True)
    fig = go.Figure()
    if choix_courbe.startswith("Taux"):
        fig.add_trace(ligne(c["t_min"], c["taux_par_heure"], "λ(t)", CATEGORIES[0]))
        styliser(fig, hauteur=280, titre_x="temps (min)", titre_y="défaillances par heure", legende=False)
    else:
        fig.add_trace(ligne(c["t_min"], c["mrl_min"], "moyenne (MRL)", CATEGORIES[0]))
        fig.add_trace(ligne(c["t_min"], c["b10_residuel_min"], "B10 conditionnel (90 %)", CATEGORIES[1]))
        styliser(fig, hauteur=280, titre_x="âge du roulement (min)", titre_y="durée de vie résiduelle (min)")
    st.plotly_chart(fig)
    if not choix_courbe.startswith("Taux"):
        st.caption("Tant qu'aucun défaut n'est détecté, la RUL est estimée par le B10 conditionnel : "
                   "la durée qu'un roulement de cet âge tiendra encore avec 90 % de chances. "
                   "La moyenne (MRL) prédit presque toujours trop tard.")

st.markdown("**Durée de vie nominale L10 des catalogues contre durées observées**")
fig = go.Figure()
for info in fiab["l10"]:
    cond = info["condition"]
    fig.add_trace(go.Scatter(x=info["vies_observees_min"], y=[f"condition {cond}"] * len(info["vies_observees_min"]),
                             mode="markers", name=f"observé — condition {cond}",
                             marker=dict(size=11, color=COULEURS_CONDITIONS[cond], line=dict(color=FOND, width=2)),
                             text=info["roulements"], hovertemplate="%{text} : %{x:.0f} min<extra></extra>"))
for info in fiab["l10"]:                       # traits L10 ajoutés après : « L10 catalogue » en fin de légende
    cond = info["condition"]
    fig.add_trace(go.Scatter(x=[info["l10_min"]], y=[f"condition {cond}"], mode="markers",
                             marker=dict(symbol="line-ns", size=26, color=ENCRE, line=dict(width=3, color=ENCRE)),
                             name="L10 catalogue", showlegend=cond == 1,
                             hovertemplate=f"L10 = {info['l10_min']:.0f} min<extra></extra>"))
styliser(fig, hauteur=260, titre_x="durée de vie (min)")
fig.update_layout(hovermode="closest")
st.plotly_chart(fig)
st.caption("L10 = (C/P)³ millions de tours (C = 4 000 N) : durée que 90 % des roulements devraient atteindre. "
           "La plupart des roulements PRONOSTIA défaillent bien avant : la loi des catalogues ne s'applique pas "
           "à ces essais accélérés, comme l'indiquent les organisateurs du challenge.")

# ---------------------------------------------------------------- vraisemblance et intervalles
comp = charger("complements.json")
st.subheader("Choix de la loi par la vraisemblance")
ic = comp["bootstrap_fiabilite"]["ic95"]
c1, c2, c3 = st.columns(3)
c1.metric("β (forme)", f"{w['beta']:.2f}", help=f"IC 95 % (bootstrap) : {ic['beta'][0]:.2f} à {ic['beta'][1]:.2f}")
c2.metric("B10", f"{w['b10_min']:.0f} min", help=f"IC 95 % : {ic['b10_min'][0]:.0f} à {ic['b10_min'][1]:.0f} min")
c3.metric("B50 (médiane)", f"{w['b50_min']:.0f} min", help=f"IC 95 % : {ic['b50_min'][0]:.0f} à {ic['b50_min'][1]:.0f} min")
st.caption(f"Intervalles de confiance à 95 % par bootstrap (on refait l'ajustement 1 000 fois en tirant les 17 "
           f"roulements au hasard) : β de {ic['beta'][0]:.2f} à {ic['beta'][1]:.2f}, B10 de {ic['b10_min'][0]:.0f} à "
           f"{ic['b10_min'][1]:.0f} min, B50 de {ic['b50_min'][0]:.0f} à {ic['b50_min'][1]:.0f} min. Avec seulement 6 "
           "défaillances, l'incertitude est grande.")

v = comp["vraisemblance"]
gauche, droite = st.columns([1.1, 1])
with gauche:
    st.markdown("**Log-vraisemblance de la loi de Weibull** (plus c'est clair, plus les données sont probables)")
    surface = v["surface"]
    fig = go.Figure()
    fig.add_trace(go.Contour(x=surface["beta"], y=surface["eta_min"], z=surface["log_vraisemblance"],
                             colorscale=[[0, "#0F1A2B"], [0.6, "#1F5FAF"], [1, "#9EC5F4"]],
                             contours=dict(start=surface["seuil_region_95"] - 12, end=surface["seuil_region_95"] + 3,
                                           size=1.5), line=dict(width=0.5, color="rgba(255, 255, 255, 0.15)"),
                             colorbar=dict(title="log-vrais.", thickness=12),
                             hovertemplate="β = %{x:.2f}, η = %{y:.0f} min : %{z:.1f}<extra></extra>"))
    fig.add_trace(go.Contour(x=surface["beta"], y=surface["eta_min"], z=surface["log_vraisemblance"],
                             contours=dict(start=surface["seuil_region_95"], end=surface["seuil_region_95"], size=1,
                                           coloring="none"), line=dict(width=2.5, color=ENCRE, dash="dash"),
                             showscale=False, name="région de confiance 95 %", hoverinfo="skip", showlegend=True))
    fig.add_trace(go.Scatter(x=[w["beta"]], y=[w["eta_min"]], mode="markers", name="maximum de vraisemblance",
                             marker=dict(size=13, color="#d95926", symbol="x", line=dict(width=1, color=FOND))))
    styliser(fig, hauteur=380, titre_x="forme β", titre_y="échelle η (min)")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig)
with droite:
    lois = pd.DataFrame(v["lois"]).drop(columns="parametres")
    st.dataframe(lois, hide_index=True, column_config={
        "loi": "Loi", "n_parametres": "Paramètres",
        "log_vraisemblance": st.column_config.NumberColumn("Log-vraisemblance", format="%.2f"),
        "aic": st.column_config.NumberColumn("AIC (plus petit = mieux)", format="%.2f")})
    lr = v["test_rapport_vraisemblance"]
    st.markdown("La vraisemblance mesure à quel point une loi explique les durées observées (en tenant compte des "
                "11 roulements censurés). L'**AIC** pénalise les lois à plus de paramètres.")
    st.markdown(f"- Les quatre lois sont proches (écart d'AIC < 2) : la log-normale est à peine meilleure. On garde "
                f"**Weibull**, la loi de référence en fiabilité, dont le paramètre β s'interprète (usure ou non).\n"
                f"- Test du rapport de vraisemblance, β = 1 contre β libre : p = {lr['p_valeur']:.3f}. Avec 6 "
                "défaillances, l'usure (β > 1) est probable mais pas prouvée au seuil de 5 %.")

# ---------------------------------------------------------------- covariables
st.subheader("Covariables : la charge change-t-elle la durée de vie ?")
cov = comp["covariables_fiabilite"]
gauche, droite = st.columns([1.1, 1])
with gauche:
    st.markdown("**Fiabilité R(t) selon la condition (modèle de Weibull à temps accéléré, covariable = charge)**")
    courbes = cov["courbes_charge"]
    fig = go.Figure()
    for c in (1, 2, 3):
        charge = {1: "4,0", 2: "4,2", 3: "5,0"}[c]
        fig.add_trace(ligne(courbes["t_min"], courbes[f"condition {c}"], f"condition {c} ({charge} kN)",
                            COULEURS_CONDITIONS[c], 2.5))
    styliser(fig, hauteur=340, titre_x="temps (min)", titre_y="fiabilité R(t)")
    st.plotly_chart(fig)
with droite:
    modeles_cov = pd.DataFrame([{
        "Covariable": m["covariable"],
        "Facteur sur la durée de vie": f"× {m['facteur_duree_de_vie']:.2f}" if "facteur_duree_de_vie" in m else "—",
        "p-valeur": f"{m['p_valeur']:.3f}" if "p_valeur" in m else "—",
        "AIC": f"{m['aic']:.1f}"} for m in cov["modeles"]])
    st.dataframe(modeles_cov, hide_index=True)
    charge = next(m for m in cov["modeles"] if m["covariable"].startswith("charge"))
    b10 = charge["b10_par_condition_min"]
    st.markdown(f"Une covariable est une variable extérieure qui peut changer la durée de vie. Ici, **+1 kN de charge "
                f"multiplie la durée de vie par {charge['facteur_duree_de_vie']:.2f}** (p = {charge['p_valeur']:.3f}). "
                f"Le B10 passe de **{b10['1']:.0f} min** (condition 1) à **{b10['3']:.0f} min** (condition 3).")
    st.caption("Attention : la vitesse baisse quand la charge monte (3 conditions seulement), donc les deux effets sont "
               "mélangés. Il faudrait plus de conditions d'essai pour les séparer.")

"""Sprint 2 — Normalisation, détection du premier défaut (FPT) et indicateur de santé (HI)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from commun import (CATEGORIES, DEGRADE, DISCRET, ENCRE, ENCRE_2, FOND, NEUTRE, charger, charger_series, entete, ligne,
                    roulements, styliser)

entete("Santé des roulements et premier défaut",
       "Chaque feature est divisée par son niveau à l'état sain (fichiers 31 à 150), puis on prend le logarithme : "
       "0 = comme à l'état sain, quelle que soit la condition opératoire.", "sante", "Sprint 2 · Santé")

sante = charger("sante.json")
rul = charger("rul.json")
series = charger_series()
tab = roulements()
seuil_hi = rul["infos_modeles"]["Wiener"]["parametres"]["seuil"]
p = sante["parametres"]

# ---------------------------------------------------------------- FPT
st.subheader("Détection du premier défaut (FPT) par la règle des 3 σ")
st.markdown(f"Alarme quand le RMS lissé (1 min) d'un axe dépasse **moyenne + {p['n_sigma']} σ** de l'état sain "
            f"pendant **{p['n_consecutifs']} mesures** de suite. L'alarme est **annulée** si le RMS repasse sous le "
            f"seuil pendant **{p['n_annulation']} mesures (5 min)** : c'était un transitoire. "
            "La règle n'utilise que le passé : elle fonctionnerait en temps réel.")

choix = st.selectbox("Roulement", tab["roulement"], index=tab["roulement"].tolist().index("Bearing2_3"))
s = series[series["roulement"] == choix]
info = sante["roulements"][choix]

c1, c2, c3 = st.columns(3)
c1.metric("FPT retenu", f"{info['fpt_min']:.1f} min")
c2.metric("Phase de dégradation", f"{info['phase_degradation_min']:.1f} min")
c3.metric("Fausses alarmes annulées", f"{len(info['episodes']) - 1}")

fig = go.Figure()
for debut, fin in info["episodes"]:
    annulee = fin < s["index"].max()
    fig.add_vrect(x0=debut * 10 / 60, x1=fin * 10 / 60, fillcolor=NEUTRE if annulee else DEGRADE, opacity=0.3,
                  line_width=0, annotation_text="" if annulee else "défaut confirmé",
                  annotation_position="top left", annotation_font_color=ENCRE_2)
fig.add_trace(ligne(s["temps_min"], s["h_rms_norm"], "RMS horizontal (lissé)", CATEGORIES[0], 1.2))
fig.add_trace(ligne(s["temps_min"], s["v_rms_norm"], "RMS vertical (lissé)", CATEGORIES[1], 1.2))
fig.add_hline(y=info["seuil_h"], line=dict(color=CATEGORIES[0], width=1, dash="dash"))
fig.add_hline(y=info["seuil_v"], line=dict(color=CATEGORIES[1], width=1, dash="dash"))
fig.add_vline(x=info["fpt_min"], line=dict(color=ENCRE, width=2))
ligne_tab = tab[tab["roulement"] == choix].iloc[0]
if ligne_tab["ensemble"] == "test":
    fig.add_vline(x=ligne_tab["troncature"] * 10 / 60, line=dict(color=ENCRE_2, width=1.5, dash="dot"),
                  annotation_text="troncature", annotation_position="bottom right")
styliser(fig, hauteur=380, titre_x="temps (min)", titre_y="log(RMS / RMS sain)")
st.plotly_chart(fig)
st.caption("Zones grises : fausses alarmes annulées ; zone orange : défaut confirmé. Pointillés horizontaux : "
           "seuils 3 σ de chaque axe. Trait blanc vertical : FPT retenu.")

# ---------------------------------------------------------------- sensibilité au seuil
with st.expander("Et si on change le seuil ? (2 à 5 σ)", icon=":material/tune:"):
    sensibilite = pd.DataFrame(charger("complements.json")["sensibilite_fpt"])
    st.dataframe(sensibilite, hide_index=True, column_config={
        "n_sigma": "Seuil (σ)", "fausses_alarmes": "Fausses alarmes (17 roulements)", "non_detectes": "Non détectés",
        "delai_alerte_median_min": st.column_config.NumberColumn("Délai d'alerte médian (min)", format="%.1f"),
        "delai_alerte_moyen_min": st.column_config.NumberColumn("Délai d'alerte moyen (min)", format="%.1f"),
        "test_degrades_a_la_troncature": "Tests dégradés à la troncature"})
    st.caption("Délai d'alerte = temps entre l'alarme retenue et la défaillance. Un seuil bas (2 σ) donne plus de "
               "fausses alarmes ; un seuil haut (4-5 σ) prévient plus tard. 3 σ est un bon compromis.")

# ---------------------------------------------------------------- classement des features
st.subheader("Sélection des features")
classement = pd.DataFrame(sante["classement"])
retenues = set(sante["selection"])
gauche, droite = st.columns([1.3, 1])
with gauche:
    t = classement.head(15).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=t["feature"], x=t["score"], orientation="h", name="score (moyenne)",
                         marker_color=[CATEGORIES[0] if f in retenues else NEUTRE for f in t["feature"]],
                         marker_cornerradius=4, hovertemplate="%{y} : %{x:.3f}<extra></extra>"))
    for colonne, nom, symbole, couleur in [("monotonie", "monotonie", "circle", ENCRE),
                                           ("correlation", "corrélation", "square", ENCRE_2),
                                           ("pronosticabilite", "pronosticabilité", "triangle-up", DISCRET)]:
        fig.add_trace(go.Scatter(y=t["feature"], x=t[colonne], mode="markers", name=nom,
                                 marker=dict(symbol=symbole, size=8, color=couleur, line=dict(color=FOND, width=1))))
    styliser(fig, hauteur=460, titre_x="critère (0 à 1)")
    fig.update_layout(hovermode="y unified")
    fig.update_xaxes(range=[0, 1])
    st.plotly_chart(fig)
with droite:
    st.markdown("Trois critères, calculés sur la phase de dégradation des 6 roulements d'apprentissage :")
    st.markdown("- **monotonie** : la feature évolue toujours dans le même sens ;\n"
                "- **corrélation** avec le temps (Spearman) : tendance régulière ;\n"
                "- **pronosticabilité** : tous les roulements finissent à peu près à la même valeur.")
    st.markdown(f"Les **{len(retenues)} meilleures** (en bleu) sont les entrées des modèles ML et deep learning. "
                "Ce sont surtout des énergies de sous-bandes d'ondelettes (wpt_k : bande k de 1,6 kHz).")
    with st.expander("Les 34 features"):
        st.dataframe(classement.round(3), hide_index=True)

# ---------------------------------------------------------------- choix du HI
st.subheader("Indicateur de santé (HI)")
choix_hi = pd.DataFrame(sante["choix_hi"])
colonnes_hi = sante["colonnes_hi"]
gauche, droite = st.columns([1.3, 1])
with gauche:
    fig = go.Figure()
    for colonne, nom, couleur in [("score", "score (moyenne)", CATEGORIES[0]), ("monotonie", "monotonie", ENCRE),
                                  ("correlation", "corrélation", ENCRE_2),
                                  ("pronosticabilite", "pronosticabilité", DISCRET)]:
        fig.add_trace(go.Scatter(x=choix_hi["k"], y=choix_hi[colonne], name=nom, mode="lines+markers",
                                 line=dict(color=couleur, width=3 if colonne == "score" else 1.5),
                                 marker=dict(size=8, line=dict(color=FOND, width=2))))
    fig.add_vline(x=len(colonnes_hi), line=dict(color=ENCRE, width=1, dash="dot"),
                  annotation_text=f"k retenu = {len(colonnes_hi)}", annotation_position="bottom right")
    styliser(fig, hauteur=320, titre_x="nombre k de features moyennées dans le HI", titre_y="critère (0 à 1)")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig)
with droite:
    st.markdown("On essaie un HI égal à la moyenne des **k meilleures features**, pour k = 1 à 10, et on garde "
                "le k qui maximise le score des trois critères sur les roulements d'apprentissage.")
    if len(colonnes_hi) == 1:
        st.markdown(f"Résultat : la **meilleure feature seule** gagne (**{colonnes_hi[0]}**). Moyenner plusieurs "
                    "features rend la fin de vie plus régulière (pronosticabilité) mais casse la tendance "
                    "(corrélation) : la sélection l'emporte ici sur la fusion.")
    else:
        st.markdown(f"Résultat : le HI est la moyenne de **{len(colonnes_hi)} features** ({', '.join(colonnes_hi)}).")
    hi = sante["criteres_hi"]
    st.dataframe(pd.DataFrame({"critère": ["monotonie", "corrélation", "pronosticabilité", "score"],
                               "HI retenu": [hi["monotonie"], hi["correlation"], hi["pronosticabilite"], hi["score"]]}),
                 hide_index=True, column_config={"HI retenu": st.column_config.NumberColumn(format="%.3f")})

# ---------------------------------------------------------------- courbes de HI
st.markdown("**Évolution du HI**")
vue = st.radio("Roulements", ["apprentissage", "test"], horizontal=True)
noms = [r for r in tab["roulement"] if (tab.set_index("roulement").loc[r, "ensemble"] == vue)]
if vue == "test":
    noms = st.multiselect("Roulements de test affichés", noms, default=["Bearing1_3", "Bearing1_4", "Bearing3_3"])
fig = go.Figure()
for i, r in enumerate(noms):
    s = series[series["roulement"] == r]
    fig.add_trace(ligne(s["temps_min"], s["hi"], r, CATEGORIES[i % len(CATEGORIES)], 1.2))
fig.add_hline(y=seuil_hi, line=dict(color=ENCRE, width=1.5, dash="dash"),
              annotation_text=f"seuil de défaillance D = {seuil_hi:.2f}", annotation_position="top left")
styliser(fig, hauteur=380, titre_x="temps (min)", titre_y="HI (0 = état sain)")
st.plotly_chart(fig)
st.caption("Le seuil D est la valeur moyenne du HI à la fin de vie des roulements d'apprentissage. "
           "Les roulements de test finissent entre des niveaux très différents : c'est une des difficultés "
           "majeures de ce jeu de données.")

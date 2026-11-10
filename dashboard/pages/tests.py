"""Sprint 4 — Tests et validation : peut-on faire confiance aux résultats ?"""
import ast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from commun import ENCRE, RACINE, charger, entete, styliser, transparent

entete("Tests et validation",
       "Avant de croire un résultat, on vérifie la méthode : sur des données simulées où la bonne réponse est "
       "connue, avec des tests statistiques, et avec des tests automatiques du code.", "tests", "Sprint 4 · Validation")

comp = charger("complements.json")
rul = charger("rul.json")
BLEUS = ["#1c5cab", "#3987e5", "#9ec5f4"]       # une seule teinte, de sombre à clair (petit -> grand échantillon)


def lister_tests():
    """Noms des fonctions de test (tests/test_*.py), lus directement dans les fichiers."""
    noms = []
    for fichier in sorted((RACINE / "tests").glob("test_*.py")):
        arbre = ast.parse(fichier.read_text(encoding="utf-8"))
        noms += [(fichier.name, n.name) for n in arbre.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    return noms


tests_code = lister_tests()

# ---------------------------------------------------------------- vue d'ensemble
st.subheader("Six façons de tester le travail")
st.table(pd.DataFrame([
    {"Type de test": "Tests unitaires (pytest)", "Question posée": "Chaque fonction calcule-t-elle juste ?",
     "Résultat": f"{len(tests_code)} fonctions de test (dont une qui ouvre chaque page du dashboard), toutes réussies"},
    {"Type de test": "Données simulées", "Question posée": "La méthode retrouve-t-elle une vérité connue ?",
     "Résultat": "Weibull retrouvée en moyenne ; intervalle du Wiener juste quand ses hypothèses sont vraies"},
    {"Type de test": "Validation croisée « un roulement de côté »",
     "Question posée": "Les hyperparamètres sont-ils choisis sans regarder le test ?",
     "Résultat": "6 tours : on apprend sur 5 roulements, on valide sur le 6e"},
    {"Type de test": "Test indépendant", "Question posée": "Le modèle marche-t-il sur des roulements jamais vus ?",
     "Résultat": "11 roulements de test, 2 protocoles (phase de dégradation, challenge)"},
    {"Type de test": "Tests statistiques", "Question posée": "Les écarts observés sont-ils réels ou dus au hasard ?",
     "Résultat": "rapport de vraisemblance, Wilcoxon, Friedman"},
    {"Type de test": "Bootstrap", "Question posée": "Quelle est l'incertitude des indicateurs ?",
     "Résultat": "intervalles à 95 % du B10, du B50, du RMSE et des scores"},
]).set_index("Type de test"))

# ---------------------------------------------------------------- données simulées : Weibull
st.subheader("Données simulées 1 : la loi de Weibull retrouve-t-elle la vérité ?")
sw = comp["simulation_weibull"]
gauche, droite = st.columns([1.2, 1])
with gauche:
    fig = go.Figure()
    for couleur, (n, info) in zip(BLEUS, sw["par_taille"].items()):
        fig.add_trace(go.Box(y=info["betas"], name=f"{n} roulements", marker_color=couleur, boxpoints=False,
                             line=dict(width=2), fillcolor=transparent(couleur, 0.35)))
    fig.add_hline(y=sw["beta_vrai"], line=dict(color=ENCRE, width=1.5, dash="dash"),
                  annotation_text=f"vrai β = {sw['beta_vrai']:.2f}", annotation_position="top left")
    styliser(fig, hauteur=360, titre_y="β estimé (400 simulations)", legende=False)
    fig.update_yaxes(range=[0.8, 4.5])
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig)
with droite:
    p17, p204 = sw["par_taille"]["17"], sw["par_taille"]["204"]
    st.markdown("On tire des durées de vie dans une loi de Weibull **connue** (β = "
                f"{sw['beta_vrai']:.2f}), avec le **même plan d'essai** que PRONOSTIA : 6 essais complets et 11 "
                "essais arrêtés aux mêmes âges. Puis on réestime β comme sur les vraies données.")
    st.markdown(f"- Avec **17 roulements**, β estimé va de **{p17['beta']['p05']} à {p17['beta']['p95']}** "
                f"(90 % des simulations) et le B10 de {p17['b10_min']['p05']:.0f} à {p17['b10_min']['p95']:.0f} min.\n"
                f"- Avec **204 roulements**, β va de {p204['beta']['p05']} à {p204['beta']['p95']}.")
    st.markdown("La méthode est juste en moyenne, mais **17 roulements, c'est peu** : nos β et B10 sont incertains.")

# ---------------------------------------------------------------- données simulées : Wiener
st.subheader("Données simulées 2 : l'intervalle du Wiener est-il fiable ?")
sim = comp["simulation_wiener"]
scenarios = [k for k in sim if k != "seuil_test_observe"]
reel = rul["couverture_wiener_90"]["moyenne"]
gauche, droite = st.columns([1.2, 1])
with gauche:
    noms = scenarios + ["vraies données (11 roulements de test)"]
    valeurs = [100 * sim[s]["couverture_90"] for s in scenarios] + [100 * reel]
    fig = go.Figure(go.Bar(x=valeurs, y=noms, orientation="h", marker_cornerradius=4,
                           marker_color=["#5598e7"] * len(scenarios) + ["#d95926"],
                           text=[f"{v:.0f} %" for v in valeurs], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y} : %{x:.0f} %<extra></extra>"))
    fig.add_vline(x=90, line=dict(color=ENCRE, width=1.5, dash="dash"), annotation_text="objectif 90 %",
                  annotation_position="top")
    styliser(fig, hauteur=320, titre_x="vraies RUL dans l'intervalle à 90 % (%)", legende=False)
    fig.update_xaxes(range=[0, 110])
    fig.update_layout(hovermode="closest", margin=dict(l=10, r=30, t=30, b=10))
    st.plotly_chart(fig)
with droite:
    s = sim["seuil_test_observe"]
    st.markdown("On simule 120 trajectoires de HI dont on connaît la vraie RUL, puis on applique notre modèle.")
    st.markdown(f"- Quand les données suivent le modèle, l'intervalle contient bien ~90 % des vraies RUL : "
                "**la méthode est correcte**.\n"
                "- Un HI qui stagne puis accélère ne la gêne presque pas.\n"
                f"- Mais si le **seuil de défaillance** varie autant que sur les roulements de test (écart-type "
                f"{s['ecart_type']:.2f} au lieu de {s['ecart_type_apprentissage']:.2f} à l'apprentissage), la "
                "couverture s'effondre, comme sur les vraies données.")
    st.markdown("**Conclusion :** le problème vient du seuil appris sur 6 roulements seulement, pas du modèle.")

# ---------------------------------------------------------------- tests statistiques
st.subheader("Tests statistiques")
ts, lr = comp["tests_statistiques"], comp["vraisemblance"]["test_rapport_vraisemblance"]
c1, c2, c3 = st.columns(3)
c1.metric("Rapport de vraisemblance (β = 1 ?)", f"p = {lr['p_valeur']:.3f}",
          help="Weibull contre exponentielle. p > 0,05 : avec 6 défaillances, on ne peut pas prouver l'usure")
c2.metric("Friedman (7 modèles)", f"p = {ts['friedman']['p_valeur']:.3f}",
          help="p < 0,05 : au moins un modèle se distingue des autres")
c3.metric("Wiener vs fiabilité seule (Wilcoxon)",
          f"p = {next(l['p_valeur'] for l in ts['wilcoxon_contre_wiener'] if l['modele'].startswith('Fiab')):.3f}")
st.dataframe(pd.DataFrame(ts["wilcoxon_contre_wiener"]), hide_index=True, column_config={
    "modele": "Modèle comparé au Wiener", "wiener_meilleur_sur": "Wiener meilleur sur (roulements sur 11)",
    "difference_mediane_min": st.column_config.NumberColumn("Écart médian de RMSE (min)", format="%.2f"),
    "p_valeur": st.column_config.NumberColumn("p-valeur (Wilcoxon)", format="%.3f"),
    "significatif_5pct": st.column_config.CheckboxColumn("Écart significatif (5 %)")})
st.caption("Test de Wilcoxon apparié sur les RMSE des 11 roulements de test. Seul l'écart avec la fiabilité seule est "
           "significatif : entre les modèles fondés sur les capteurs, 11 roulements ne suffisent pas à départager.")

# ---------------------------------------------------------------- tests du code
st.subheader("Tests automatiques du code")
st.markdown(f"**{len(tests_code)} fonctions de test**, lancées avec `python -m pytest`. Chacune compare le code à une "
            "réponse connue à la main (score parfait, RMS d'une sinusoïde, kurtosis d'un bruit gaussien = 3, Weibull "
            "avec β = 1, coût d'une stratégie simple...).")
st.dataframe(pd.DataFrame([{"Fichier": f, "Test": n.replace("test_", "").replace("_", " ")} for f, n in tests_code]),
             hide_index=True, height=320)
st.caption("Graine aléatoire fixée (42) : les résultats se refont à l'identique d'une exécution à l'autre.")

"""Dashboard du projet — lancer depuis le dossier du projet :

    streamlit run dashboard/app.py

Il lit les résultats déjà calculés (dossier results/) : pas besoin de relancer les calculs.
"""
import streamlit as st

st.set_page_config(page_title="Pronostic de roulements — PRONOSTIA", page_icon="〽️", layout="wide")

pages = [
    st.Page("pages/accueil.py", title="Vue d'ensemble", icon=":material/dashboard:", default=True),
    st.Page("pages/donnees.py", title="1 · Données et fiabilité", icon=":material/vibration:"),
    st.Page("pages/sante.py", title="2 · Santé et FPT", icon=":material/monitor_heart:"),
    st.Page("pages/pronostic.py", title="3 · Pronostic de la RUL", icon=":material/timer:"),
    st.Page("pages/comparaison.py", title="4 · Comparaison des modèles", icon=":material/leaderboard:"),
    st.Page("pages/economie.py", title="5 · Décision et économie", icon=":material/savings:"),
    st.Page("pages/tests.py", title="6 · Tests et validation", icon=":material/fact_check:"),
]

with st.sidebar:
    st.markdown("### Pronostic de roulements")
    st.caption("Étude comparative de pronostic de défaillance - données FEMTO-ST / PRONOSTIA (IEEE PHM 2012)")
    st.html('<div class="signature">Projet de Fiabilité et Maintenance<br>ENSIT · GMAM 3 · 2026–2027</div>')

st.navigation(pages).run()

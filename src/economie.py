"""Sprint 4 — Décision de maintenance et analyse économique.

On compare plusieurs stratégies sur les 11 roulements de test, dont on connaît la vraie durée de vie :
- corrective    : on attend la panne (coût Cc) ;
- préventive    : on remplace à un âge fixe T*, choisi avec la loi de Weibull (coût Cp) ;
- conditionnelle: on remplace dès que le défaut est détecté (FPT), après le délai d'intervention ;
- prédictive    : on remplace quand la RUL prédite par un modèle descend sous un seuil s ;
- idéale        : on remplace juste avant la panne (référence que personne ne peut battre).

Coût par heure = coût total / temps de fonctionnement total (on remplace, on recommence : processus
de renouvellement). Les coûts sont en unités relatives : 1 u = le coût d'un remplacement planifié.
Une fausse alarme coûte une inspection (Ci) pour les stratégies qui surveillent le roulement.
"""
import numpy as np

PARAMETRES = {"cp": 1.0, "cc": 5.0, "ci": 0.1, "delai_min": 2.0, "seuil_min": 10.0}


# ---------------------------------------------------------------- préventive : âge optimal T*
def fiabilite(t, beta, eta):
    """Loi de Weibull R(t) = exp(-(t/eta)^beta) (même formule que src/reliability.py, recopiée ici
    pour que le dashboard n'ait besoin que de numpy)."""
    return np.exp(-(np.asarray(t, float) / eta) ** beta)


def couts_age(beta, eta, cp, cc, t_max=1000.0, pas=0.5):
    """Coût par heure de la stratégie « remplacer à l'âge T », pour T de 0,5 à t_max minutes :
    C(T) = [Cp R(T) + Cc (1 - R(T))] / (intégrale de R entre 0 et T)   (formule du cours)."""
    ages = np.arange(pas, t_max + pas, pas)
    r = fiabilite(ages, beta, eta)
    duree_moyenne = np.cumsum(r) * pas                    # ≈ intégrale de R(u) entre 0 et T
    return ages, (cp * r + cc * (1 - r)) / duree_moyenne * 60


def age_optimal(beta, eta, cp, cc):
    """Âge T* qui minimise le coût par heure, et ce coût."""
    ages, couts = couts_age(beta, eta, cp, cc)
    k = int(np.argmin(couts))
    return float(ages[k]), float(couts[k])


# ---------------------------------------------------------------- décisions
def instant_declenchement(temps, rul_predite, seuil):
    """Premier instant où la RUL prédite passe sous le seuil (None si jamais)."""
    sous_le_seuil = np.flatnonzero(np.asarray(rul_predite) <= seuil)
    if len(sous_le_seuil) == 0:
        return None
    return float(temps[sous_le_seuil[0]])


def evaluer(remplacements, vies, cp, cc, ci=0.0, fausses_alarmes=None):
    """Bilan d'une stratégie. remplacements[i] = instant prévu du remplacement (min) ou None ;
    vies[i] = vraie durée de vie (min). Remplacé avant la panne -> Cp, sinon panne -> Cc."""
    couts, utilisees, pannes, perdues = [], [], 0, []
    for i, (t_remplacement, vie) in enumerate(zip(remplacements, vies)):
        # arrondi au centième de minute (précision des temps enregistrés) : un remplacement prévu
        # pile à l'instant de la panne compte comme une panne
        if t_remplacement is not None and round(t_remplacement, 2) < round(vie, 2):
            couts.append(cp)
            utilisees.append(t_remplacement)
            perdues.append(vie - t_remplacement)        # vie gaspillée : le roulement aurait encore tenu
        else:
            couts.append(cc)
            utilisees.append(vie)
            pannes += 1
        if fausses_alarmes is not None:
            couts[-1] += ci * fausses_alarmes[i]
    return {
        "cout_total_u": round(float(sum(couts)), 2),
        "cout_par_heure_u": round(float(sum(couts) / (sum(utilisees) / 60)), 4),
        "pannes": int(pannes),
        "pannes_evitees_pct": round(100 * (1 - pannes / len(vies)), 1),
        "vie_perdue_moyenne_min": round(float(np.mean(perdues)), 1) if perdues else 0.0,
    }


def comparer_strategies(vies, fpt, courbes, fausses_alarmes, beta, eta, cp=1.0, cc=5.0, ci=0.1,
                        delai_min=2.0, seuil_min=10.0, modeles=()):
    """Toutes les stratégies sur les mêmes roulements. vies, fpt, fausses_alarmes : un nombre par roulement ;
    courbes : liste de dictionnaires {temps_min, predictions{modele: RUL}, wiener_bas} par roulement."""
    t_etoile, _ = age_optimal(beta, eta, cp, cc)
    n = len(vies)
    resultats = {
        "Corrective": evaluer([None] * n, vies, cp, cc),
        "Préventive (âge T*)": evaluer([t_etoile] * n, vies, cp, cc),
        "Conditionnelle (dès le FPT)": evaluer([f + delai_min for f in fpt], vies, cp, cc, ci, fausses_alarmes),
    }
    for modele in modeles:
        remplacements = []
        for courbe in courbes:
            t = instant_declenchement(courbe["temps_min"], courbe["predictions"][modele], seuil_min)
            remplacements.append(None if t is None else t + delai_min)
        resultats[f"Prédictive ({modele})"] = evaluer(remplacements, vies, cp, cc, ci, fausses_alarmes)
    remplacements = []                                  # décision prudente : borne basse (5 %) du Wiener
    for courbe in courbes:
        t = instant_declenchement(courbe["temps_min"], courbe["wiener_bas"], seuil_min)
        remplacements.append(None if t is None else t + delai_min)
    resultats["Prédictive (Wiener prudent, 5 %)"] = evaluer(remplacements, vies, cp, cc, ci, fausses_alarmes)
    resultats["Idéale (juste avant la panne)"] = evaluer([v - 0.01 for v in vies], vies, cp, cc)

    reference = resultats["Corrective"]["cout_par_heure_u"]
    for bilan in resultats.values():
        bilan["gain_vs_corrective_pct"] = round(100 * (1 - bilan["cout_par_heure_u"] / reference), 1)
    return resultats, t_etoile

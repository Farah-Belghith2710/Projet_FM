"""Sprint 1 (analyse préliminaire) — Fiabilité de la population de roulements.

Données de durée de vie (ce qu'on savait au moment du challenge) :
- 6 roulements d'apprentissage : défaillance observée (événement = 1) ;
- 11 roulements de test : encore en fonctionnement à la troncature -> censurés à droite (événement = 0).

On calcule : Kaplan-Meier, loi de Weibull par maximum de vraisemblance (avec censure), MTTF, B10,
taux de défaillance, durée de vie théorique L10 des catalogues, et la durée de vie résiduelle
d'un roulement encore sain (utilisée tant qu'aucun défaut n'est détecté) : moyenne (MRL)
ou à 90 % de fiabilité conditionnelle (B10 conditionnel).
"""
import numpy as np
from lifelines import KaplanMeierFitter, WeibullFitter
from scipy.integrate import quad
from scipy.special import gamma

from src import config


def donnees_survie(roulements):
    """Durées (en minutes) et événements (1 = défaillance, 0 = censure) à partir du tableau des roulements."""
    durees, evenements = [], []
    for _, ligne in roulements.iterrows():
        if ligne["ensemble"] == "apprentissage":
            durees.append(ligne["duree_vie_s"] / 60)
            evenements.append(1)
        else:
            durees.append(ligne["troncature"] * config.PAS_TEMPS / 60)
            evenements.append(0)
    return np.array(durees), np.array(evenements)


def kaplan_meier(durees, evenements):
    """Courbe de survie empirique avec son intervalle de confiance à 95 %."""
    km = KaplanMeierFitter()
    km.fit(durees, event_observed=evenements)
    ic = km.confidence_interval_
    return {
        "t": km.survival_function_.index.tolist(),
        "survie": km.survival_function_.iloc[:, 0].tolist(),
        "bas": ic.iloc[:, 0].tolist(),
        "haut": ic.iloc[:, 1].tolist(),
    }


def ajuster_weibull(durees, evenements):
    """Weibull à 2 paramètres par maximum de vraisemblance : R(t) = exp(-(t/eta)^beta)."""
    wf = WeibullFitter()
    wf.fit(durees, event_observed=evenements)
    return float(wf.rho_), float(wf.lambda_)        # beta (forme), eta (échelle, en minutes)


def fiabilite(t, beta, eta):
    return np.exp(-(np.asarray(t) / eta) ** beta)


def taux_defaillance(t, beta, eta):
    t = np.asarray(t, float)
    return beta / eta * (t / eta) ** (beta - 1)


def mttf(beta, eta):
    return eta * gamma(1 + 1 / beta)


def quantile_vie(p, beta, eta):
    """Âge auquel une proportion p de la population a défailli (p = 0,10 -> B10)."""
    return eta * (-np.log(1 - p)) ** (1 / beta)


def l10_catalogue(condition):
    """Durée de vie nominale des catalogues (minutes) : L10 = (C/P)^3 millions de tours."""
    vitesse = config.CONDITIONS[condition]["vitesse_rpm"]
    charge = config.CONDITIONS[condition]["charge_N"]
    millions_de_tours = (config.CHARGE_DYNAMIQUE_C / charge) ** 3
    return millions_de_tours * 1e6 / vitesse


def vie_residuelle_b10(age, beta, eta):
    """B10 conditionnel : durée x telle que R(age + x) / R(age) = 0,90.
    Avec 90 % de chances, un roulement qui fonctionne encore à 'age' tiendra encore x minutes.
    C'est la même convention que la L10 des catalogues (fiabilité de 90 %), et une prédiction
    prudente : le score du challenge pénalise beaucoup plus les prédictions trop tardives."""
    return eta * ((age / eta) ** beta - np.log(0.90)) ** (1 / beta) - age


def vie_residuelle_moyenne(age, beta, eta):
    """MRL(t) = (intégrale de R(u) entre t et l'infini) / R(t) : durée de vie restante moyenne
    d'un roulement qui fonctionne encore à l'âge t (en minutes)."""
    integrale = quad(fiabilite, age, np.inf, args=(beta, eta))[0]
    return integrale / fiabilite(age, beta, eta)

"""Sprint 4 — Métriques d'évaluation PHM.

- RMSE et MAE sur toute la trajectoire de RUL (en minutes) ;
- erreur en pourcentage et score asymétrique du challenge IEEE PHM 2012 :
      %Er = 100 x (RUL_réelle - RUL_prédite) / RUL_réelle
      A   = exp(-ln(0,5) x %Er / 5)   si %Er <= 0  (prédiction tardive : forte pénalité)
      A   = exp(+ln(0,5) x %Er / 20)  si %Er > 0   (prédiction précoce : pénalité plus douce)
      Score = moyenne des A sur les 11 roulements de test (1 = parfait)
- intervalle de confiance du score par bootstrap (rééchantillonnage des roulements).
"""
import numpy as np


def rmse(vrai, predit):
    vrai, predit = np.asarray(vrai, float), np.asarray(predit, float)
    return float(np.sqrt(np.mean((predit - vrai) ** 2)))


def mae(vrai, predit):
    vrai, predit = np.asarray(vrai, float), np.asarray(predit, float)
    return float(np.mean(np.abs(predit - vrai)))


def erreur_pourcentage(rul_reelle, rul_predite):
    """%Er > 0 : prédiction trop tôt (prudente) ; %Er < 0 : prédiction trop tard (dangereuse)."""
    return 100 * (rul_reelle - rul_predite) / rul_reelle


def score_phm(erreur):
    """Score A d'une prédiction à partir de son %Er (formule (2) du challenge)."""
    erreur = np.clip(np.asarray(erreur, float), -1000, 1000)   # au-delà, le score vaut 0 de toute façon
    return np.where(erreur <= 0,
                    np.exp(-np.log(0.5) * erreur / 5),
                    np.exp(np.log(0.5) * erreur / 20))


def bootstrap_ic(valeurs, n_tirages=2000, niveau=0.95, graine=0):
    """Intervalle de confiance de la moyenne par bootstrap : on retire les roulements au hasard,
    avec remise, et on regarde comment la moyenne varie."""
    valeurs = np.asarray(valeurs, float)
    rng = np.random.default_rng(graine)
    tirages = rng.choice(valeurs, size=(n_tirages, len(valeurs)), replace=True)
    moyennes = tirages.mean(axis=1)
    alpha = (1 - niveau) / 2
    return float(np.quantile(moyennes, alpha)), float(np.quantile(moyennes, 1 - alpha))

"""Sprint 3 — Machine learning classique : SVR, GPR (processus gaussien) et Random Forest.

Entrées à un instant t après le FPT : les features du HI (normalisées, lissées)
+ le temps écoulé depuis le FPT (en minutes).
Cible : log(1 + RUL en minutes). Avec le logarithme, se tromper de 1 min sur une RUL de 2 min
compte autant que se tromper de 30 min sur une RUL de 60 min : c'est l'esprit du score du
challenge, qui juge l'erreur en pourcentage.

Chaque roulement d'apprentissage donne le même nombre de points (100, tirés au hasard),
sinon les roulements à dégradation lente (des centaines de points) écraseraient les autres.
Les hyperparamètres sont choisis par validation croisée « un roulement de côté »
(LeaveOneGroupOut) : à chaque tour, on apprend sur 5 roulements et on teste sur le 6e.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src import config

N_PAR_ROULEMENT = 100


def ajouter_depuis_fpt(lisse):
    """Colonne 'depuis_fpt_min' : minutes écoulées depuis le FPT final (NaN avant le FPT)."""
    lisse = lisse.copy()
    depuis = (lisse["index"] - lisse["fpt"]) * config.PAS_TEMPS / 60
    lisse["depuis_fpt_min"] = depuis.where(depuis >= 0)
    return lisse


def jeu_apprentissage(lisse, roulements, entrees, graine=config.GRAINE):
    """X, y (log RUL) et groupes (nom du roulement), avec 100 points par roulement."""
    rng = np.random.default_rng(graine)
    morceaux = []
    for roulement in roulements:
        phase = lisse[(lisse["roulement"] == roulement) & (lisse["depuis_fpt_min"] >= 0)]
        avec_remise = len(phase) < N_PAR_ROULEMENT      # roulements courts : on réutilise des points
        choix = rng.choice(len(phase), N_PAR_ROULEMENT, replace=avec_remise)
        morceaux.append(phase.iloc[np.sort(choix)])
    jeu = pd.concat(morceaux)
    X = jeu[entrees].to_numpy()
    y = np.log1p(jeu["rul_s"].to_numpy() / 60)
    return X, y, jeu["roulement"].to_numpy()


def modeles_et_grilles():
    """Les trois modèles et les hyperparamètres testés pour chacun."""
    noyau = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    return {
        "SVR": (
            make_pipeline(StandardScaler(), SVR()),
            {"svr__C": [0.3, 1, 3, 10], "svr__gamma": ["scale", 0.03, 0.1], "svr__epsilon": [0.05, 0.2]},
        ),
        "GPR": (
            make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=noyau, normalize_y=True,
                                                                       random_state=config.GRAINE)),
            # les paramètres du noyau sont optimisés automatiquement (vraisemblance marginale) ;
            # on teste seulement le point de départ de la longueur de corrélation
            {"gaussianprocessregressor__kernel__k1__k2__length_scale": [0.3, 1.0, 3.0]},
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=300, random_state=config.GRAINE, n_jobs=-1),
            {"max_depth": [4, 8, None], "min_samples_leaf": [1, 5, 20], "max_features": [0.5, 1.0]},
        ),
    }


def entrainer(nom, X, y, groupes):
    """Recherche sur grille + validation « un roulement de côté ». Renvoie le meilleur modèle."""
    modele, grille = modeles_et_grilles()[nom]
    debut = time.time()
    recherche = GridSearchCV(modele, grille, cv=LeaveOneGroupOut(),
                             scoring="neg_mean_absolute_error", n_jobs=-1)
    recherche.fit(X, y, groups=groupes)
    infos = {
        "meilleurs_parametres": {k.split("__")[-1]: v for k, v in recherche.best_params_.items()},
        "mae_validation_log": float(-recherche.best_score_),
        "n_combinaisons": len(recherche.cv_results_["params"]),
        "duree_s": round(time.time() - debut, 1),
    }
    return recherche.best_estimator_, infos


def predire(modele, X):
    """Prédiction en minutes (on revient de log(1 + RUL) à la RUL)."""
    return np.maximum(np.expm1(modele.predict(X)), 0)

"""Sprint 3 — Deep learning : LSTM et réseau convolutif 1D (CNN 1D), avec Keras.

Au lieu d'un seul instant, le réseau voit une fenêtre des dernières mesures (par exemple
12 mesures = 2 minutes) : il peut donc exploiter la dynamique de la dégradation.
Entrées de chaque pas de temps : les features du HI + le temps relatif au FPT (en minutes).
Cible : log(1 + RUL en minutes), comme pour le machine learning classique.

- LSTM : réseau récurrent qui lit la fenêtre pas à pas et garde une « mémoire ».
- CNN 1D : filtres convolutifs qui glissent le long du temps et repèrent des motifs locaux.

Hyperparamètres (taille du réseau et longueur de la fenêtre) choisis par la même validation
« un roulement de côté » que pour le machine learning classique.
"""
import logging
import time

import keras
import numpy as np
from keras import layers

from src import config

logging.getLogger("tensorflow").setLevel(logging.ERROR)   # cache les messages techniques de TensorFlow

N_PAR_ROULEMENT = 100
EPOQUES = 80
GRILLE = {"unites": [16, 32], "fenetre": [6, 12]}


def fenetres_roulement(groupe, entrees, fenetre):
    """Pour un roulement : une fenêtre (fenetre x nb_entrees) pour chaque instant après le FPT,
    et la RUL (minutes) correspondante."""
    groupe = groupe.reset_index(drop=True)
    valeurs = groupe[entrees].to_numpy()
    fpt = groupe["fpt"].iloc[0]
    temps_relatif = ((groupe["index"] - fpt) * config.PAS_TEMPS / 60).to_numpy()
    valeurs = np.column_stack([valeurs, temps_relatif])

    positions = np.where(groupe["index"] >= fpt)[0]
    X = np.array([valeurs[p - fenetre + 1: p + 1] for p in positions])
    rul = groupe["rul_s"].to_numpy()[positions] / 60
    return X, rul


def jeu_apprentissage(lisse, roulements, entrees, fenetre, graine=config.GRAINE):
    """Fenêtres d'apprentissage : 100 par roulement (comme pour le machine learning classique)."""
    rng = np.random.default_rng(graine)
    tous_X, tous_y, groupes = [], [], []
    for roulement in roulements:
        X, rul = fenetres_roulement(lisse[lisse["roulement"] == roulement], entrees, fenetre)
        choix = rng.choice(len(X), N_PAR_ROULEMENT, replace=len(X) < N_PAR_ROULEMENT)
        tous_X.append(X[choix])
        tous_y.append(np.log1p(rul[choix]))
        groupes += [roulement] * N_PAR_ROULEMENT
    return np.concatenate(tous_X), np.concatenate(tous_y), np.array(groupes)


def creer_reseau(type_reseau, fenetre, n_entrees, unites):
    """Petit réseau LSTM ou CNN 1D (quelques milliers de paramètres : on a peu de données)."""
    if type_reseau == "LSTM":
        couches = [layers.LSTM(unites), layers.Dropout(0.2)]
    else:
        couches = [layers.Conv1D(unites, 3, activation="relu", padding="same"),
                   layers.Conv1D(unites, 3, activation="relu", padding="same"),
                   layers.GlobalAveragePooling1D(), layers.Dropout(0.2)]
    reseau = keras.Sequential([layers.Input((fenetre, n_entrees))] + couches +
                              [layers.Dense(16, activation="relu"), layers.Dense(1)])
    reseau.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return reseau


class ReseauRUL:
    """Réseau + normalisation des entrées (moyenne et écart-type de l'apprentissage)."""

    def __init__(self, type_reseau, fenetre, unites):
        self.type_reseau, self.fenetre, self.unites = type_reseau, fenetre, unites

    def fit(self, X, y):
        keras.utils.set_random_seed(config.GRAINE)
        self.moyenne = X.reshape(-1, X.shape[2]).mean(axis=0)
        self.ecart = X.reshape(-1, X.shape[2]).std(axis=0) + 1e-8
        self.reseau = creer_reseau(self.type_reseau, self.fenetre, X.shape[2], self.unites)
        self.historique = self.reseau.fit((X - self.moyenne) / self.ecart, y, epochs=EPOQUES,
                                          batch_size=32, verbose=0).history["loss"]
        return self

    def predict(self, X):
        return self.reseau.predict((X - self.moyenne) / self.ecart, verbose=0).ravel()


def entrainer(type_reseau, lisse, roulements, entrees, recherche=True):
    """Choisit (unites, fenetre) par validation « un roulement de côté », puis réentraîne sur tout."""
    debut = time.time()
    resultats = []
    combinaisons = [(u, f) for u in GRILLE["unites"] for f in GRILLE["fenetre"]]
    if not recherche:
        combinaisons = combinaisons[-1:]

    for unites, fenetre in combinaisons:
        X, y, groupes = jeu_apprentissage(lisse, roulements, entrees, fenetre)
        erreurs = []
        if recherche:
            for roulement in roulements:                 # un roulement de côté à chaque tour
                test = groupes == roulement
                modele = ReseauRUL(type_reseau, fenetre, unites).fit(X[~test], y[~test])
                erreurs.append(np.mean(np.abs(modele.predict(X[test]) - y[test])))
        resultats.append({"unites": unites, "fenetre": fenetre,
                          "mae_validation_log": float(np.mean(erreurs)) if erreurs else None})

    meilleur = min(resultats, key=lambda r: r["mae_validation_log"] if r["mae_validation_log"] else 0)
    X, y, _ = jeu_apprentissage(lisse, roulements, entrees, meilleur["fenetre"])
    modele = ReseauRUL(type_reseau, meilleur["fenetre"], meilleur["unites"]).fit(X, y)

    infos = {"meilleurs_parametres": {"unites": meilleur["unites"], "fenetre": meilleur["fenetre"]},
             "mae_validation_log": meilleur["mae_validation_log"], "grille": resultats,
             "n_combinaisons": len(combinaisons), "courbe_apprentissage": [float(v) for v in modele.historique],
             "n_parametres": int(modele.reseau.count_params()), "duree_s": round(time.time() - debut, 1)}
    return modele, infos


def predire(modele, groupe, entrees):
    """RUL prédite (minutes) à chaque instant après le FPT d'un roulement."""
    X, _ = fenetres_roulement(groupe, entrees, modele.fenetre)
    return np.maximum(np.expm1(modele.predict(X)), 0)

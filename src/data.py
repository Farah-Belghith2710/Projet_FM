"""Sprint 1 — Chargement des signaux vibratoires FEMTO / PRONOSTIA.

Chaque roulement est un dossier de fichiers acc_00001.csv, acc_00002.csv, ...
Un fichier = 2560 points (0,1 s) enregistrés toutes les 10 s, avec 6 colonnes :
heure, minute, seconde, microseconde, accélération horizontale, accélération verticale (en g).
"""
import numpy as np
import pandas as pd

from src import config


def lister_fichiers(roulement):
    """Liste triée des fichiers de vibration d'un roulement."""
    return sorted(config.dossier_de(roulement).glob("acc_*.csv"))


def lire_fichier(chemin):
    """Lit un fichier acc_xxxxx.csv et renvoie (horizontal, vertical, heure_en_secondes).

    Attention : les fichiers de Bearing1_4 utilisent ';' comme séparateur, les autres ','.
    """
    with open(chemin) as f:
        premiere_ligne = f.readline()
    separateur = ";" if ";" in premiere_ligne else ","

    tableau = pd.read_csv(chemin, header=None, sep=separateur).to_numpy()
    heure = tableau[0, 0] * 3600 + tableau[0, 1] * 60 + tableau[0, 2] + tableau[0, 3] * 1e-6
    return tableau[:, 4], tableau[:, 5], heure


def tableau_roulements():
    """Une ligne par roulement : condition, nombre de fichiers, durée de vie, troncature, RUL officielle."""
    lignes = []
    for roulement in config.TOUS:
        condition = config.condition_de(roulement)
        n_fichiers = len(lister_fichiers(roulement))
        ligne = {
            "roulement": roulement,
            "ensemble": "apprentissage" if roulement in config.APPRENTISSAGE else "test",
            "condition": condition,
            "vitesse_rpm": config.CONDITIONS[condition]["vitesse_rpm"],
            "charge_N": config.CONDITIONS[condition]["charge_N"],
            "n_fichiers": n_fichiers,
            "duree_vie_s": n_fichiers * config.PAS_TEMPS,
            "troncature": None,
            "rul_officielle_s": None,
        }
        if roulement in config.TEST:
            ligne["troncature"] = config.TEST[roulement]["troncature"]
            ligne["rul_officielle_s"] = config.TEST[roulement]["rul_reelle"]
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def controle_qualite(signaux_h, signaux_v, heures):
    """Petits contrôles sur un roulement (Sprint 1 : valeurs manquantes, aberrations).

    signaux_h, signaux_v : listes de tableaux (un par fichier) ; heures : liste des heures de début.
    """
    longueurs = np.array([len(x) for x in signaux_h])
    n_nan = sum(int(np.isnan(h).sum() + np.isnan(v).sum()) for h, v in zip(signaux_h, signaux_v))

    # écart entre deux fichiers : doit valoir 10 s (on corrige le passage à minuit)
    ecarts = np.diff(np.array(heures))
    ecarts[ecarts < -40000] += 86400
    n_horodatages_anormaux = int(np.sum(np.abs(ecarts - config.PAS_TEMPS) > 1))

    return {
        "fichiers_mauvaise_longueur": int(np.sum(longueurs != config.N_POINTS)),
        "valeurs_manquantes": n_nan,
        "horodatages_anormaux": n_horodatages_anormaux,
    }

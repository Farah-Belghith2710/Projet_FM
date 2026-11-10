"""ÉTAPE 1 (Sprints 1 et 2) — Signaux bruts -> table de features.

Pour chaque roulement et chaque fichier de 0,1 s :
lecture -> contrôle qualité -> suppression de la dérive + filtre passe-bande -> features.

C'est la seule étape qui a besoin des données brutes (~2 Go, dossier data/raw, voir le README).
Elle produit :
    data/processed/features.csv    une ligne par fichier (24 889 lignes)
    data/processed/roulements.csv  une ligne par roulement (durée de vie, troncature, qualité)
    data/processed/signaux.json    quelques signaux bruts et filtrés, pour le dashboard

Durée : 3 à 10 minutes.      Lancer :  python 1_extraire_features.py
"""
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.data import controle_qualite, lire_fichier, lister_fichiers, tableau_roulements
from src.features import extraire
from src.preprocessing import nettoyer


def traiter_roulement(roulement):
    fichiers = lister_fichiers(roulement)
    n = len(fichiers)
    lignes, bruts_h, bruts_v, heures = [], [], [], []

    # 3 instants gardés pour le dashboard : début, milieu et fin de vie
    instants = [0, n // 2, n - 1]
    signaux = {"instants_s": [], "brut_h": [], "filtre_h": []}

    for k, chemin in enumerate(fichiers):
        h, v, heure = lire_fichier(chemin)
        bruts_h.append(h)
        bruts_v.append(v)
        heures.append(heure)

        h_propre = nettoyer(h)
        v_propre = nettoyer(v)

        ligne = {
            "roulement": roulement,
            "index": k + 1,
            "temps_s": (k + 1) * config.PAS_TEMPS,
            "h_max": h.max(), "h_min": h.min(),     # amplitudes brutes (critère des 20 g)
            "v_max": v.max(), "v_min": v.min(),
        }
        ligne.update(extraire(h_propre, "h_"))
        ligne.update(extraire(v_propre, "v_"))
        lignes.append(ligne)

        if k in instants:
            signaux["instants_s"].append((k + 1) * config.PAS_TEMPS)
            signaux["brut_h"].append(np.round(h, 3).tolist())
            signaux["filtre_h"].append(np.round(h_propre, 3).tolist())

    qualite = controle_qualite(bruts_h, bruts_v, heures)
    return pd.DataFrame(lignes), qualite, signaux


def main():
    if not config.DOSSIER_BRUT.exists():
        print("Données brutes introuvables dans", config.DOSSIER_BRUT)
        print("Voir le README (section « Repartir des fichiers bruts ») pour les télécharger.")
        return

    config.DOSSIER_TRAITE.mkdir(parents=True, exist_ok=True)
    roulements = tableau_roulements()
    tables, qualites, tous_signaux = [], [], {}
    debut = time.time()

    for roulement in config.TOUS:
        t0 = time.time()
        table, qualite, signaux = traiter_roulement(roulement)
        tables.append(table)
        qualites.append(qualite)
        tous_signaux[roulement] = signaux
        print(f"{roulement:12s} {len(table):5d} fichiers   {time.time() - t0:5.1f} s")

    features = pd.concat(tables, ignore_index=True)

    # RUL vraie de chaque fichier (durée de vie - temps écoulé), en secondes
    duree = features.groupby("roulement")["temps_s"].transform("max")
    features["rul_s"] = duree - features["temps_s"]

    # Premier dépassement de 20 g (critère de fin d'essai du challenge)
    amplitude = features[["h_max", "h_min", "v_max", "v_min"]].abs().max(axis=1)
    depasse = features[amplitude > config.SEUIL_DEFAILLANCE_G]
    premier = depasse.groupby("roulement")["temps_s"].min()
    roulements["premier_depassement_20g_s"] = roulements["roulement"].map(premier)

    roulements = pd.concat([roulements, pd.DataFrame(qualites)], axis=1)

    features.round(6).to_csv(config.FICHIER_FEATURES, index=False)
    roulements.to_csv(config.DOSSIER_TRAITE / "roulements.csv", index=False)
    with open(config.DOSSIER_TRAITE / "signaux.json", "w") as f:
        json.dump(tous_signaux, f)

    print(f"\nTerminé en {time.time() - debut:.0f} s -> {config.FICHIER_FEATURES}")
    print(roulements.to_string(index=False))


if __name__ == "__main__":
    main()

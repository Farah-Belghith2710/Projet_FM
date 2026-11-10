"""Paramètres du projet : chemins, roulements, conditions opératoires.

Toutes les constantes sont regroupées ici pour ne les écrire qu'une seule fois.
Source des valeurs : IEEE PHM 2012 Prognostic Challenge (FEMTO-ST, plateforme PRONOSTIA).
"""
from pathlib import Path

# ---------------------------------------------------------------- chemins
RACINE = Path(__file__).resolve().parents[1]
DOSSIER_BRUT = RACINE / "data" / "raw"            # Learning_set/ et Full_Test_Set/ (non fournis, voir README)
DOSSIER_TRAITE = RACINE / "data" / "processed"    # features.csv (fourni)
DOSSIER_RESULTATS = RACINE / "results"            # JSON lus par le dashboard
DOSSIER_FIGURES = RACINE / "figures"              # figures PNG du rapport

FICHIER_FEATURES = DOSSIER_TRAITE / "features.csv"

# ---------------------------------------------------------------- acquisition
FS = 25600              # fréquence d'échantillonnage des accéléromètres (Hz)
N_POINTS = 2560         # points par fichier (= 0,1 s de signal)
PAS_TEMPS = 10          # un fichier toutes les 10 secondes
SEUIL_DEFAILLANCE_G = 20  # critère d'arrêt de l'essai : vibration > 20 g

# ---------------------------------------------------------------- conditions opératoires
CONDITIONS = {
    1: {"vitesse_rpm": 1800, "charge_N": 4000},
    2: {"vitesse_rpm": 1650, "charge_N": 4200},
    3: {"vitesse_rpm": 1500, "charge_N": 5000},
}

# Roulement testé : charge dynamique de base C = 4000 N (annexe A.1 du challenge)
CHARGE_DYNAMIQUE_C = 4000

# ---------------------------------------------------------------- roulements
# 6 roulements d'apprentissage (essais complets jusqu'à la défaillance)
APPRENTISSAGE = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]

# 11 roulements de test : données tronquées pendant le challenge.
#  - "troncature" : nombre de fichiers donnés aux participants (dossier Test_set)
#  - "rul_reelle" : RUL officielle à prédire, en secondes (tableau 3 du challenge)
TEST = {
    "Bearing1_3": {"troncature": 1802, "rul_reelle": 5730},
    "Bearing1_4": {"troncature": 1139, "rul_reelle": 339},
    "Bearing1_5": {"troncature": 2302, "rul_reelle": 1610},
    "Bearing1_6": {"troncature": 2302, "rul_reelle": 1460},
    "Bearing1_7": {"troncature": 1502, "rul_reelle": 7570},
    "Bearing2_3": {"troncature": 1202, "rul_reelle": 7530},
    "Bearing2_4": {"troncature": 612, "rul_reelle": 1390},
    "Bearing2_5": {"troncature": 2002, "rul_reelle": 3090},
    "Bearing2_6": {"troncature": 572, "rul_reelle": 1290},
    "Bearing2_7": {"troncature": 172, "rul_reelle": 580},
    "Bearing3_3": {"troncature": 352, "rul_reelle": 820},
}

TOUS = APPRENTISSAGE + list(TEST)


def condition_de(roulement):
    """'Bearing2_5' -> 2 (le premier chiffre donne la condition opératoire)."""
    return int(roulement[7])


def dossier_de(roulement):
    """Dossier des fichiers bruts : Learning_set pour l'apprentissage, Full_Test_Set pour le test."""
    if roulement in APPRENTISSAGE:
        return DOSSIER_BRUT / "Learning_set" / roulement
    return DOSSIER_BRUT / "Full_Test_Set" / roulement


# ---------------------------------------------------------------- graine aléatoire
GRAINE = 42

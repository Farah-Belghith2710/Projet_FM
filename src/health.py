"""Sprint 2 — Normalisation, détection du premier défaut (FPT) et indicateur de santé (HI).

1. Normalisation multi-conditions : chaque feature est divisée par son niveau à l'état sain
   (moyenne des fichiers 31 à 150, soit de 5 à 25 min ; on ignore les 5 premières minutes de rodage),
   puis on prend le logarithme. Une valeur 0 veut dire « comme à l'état sain », quelle que soit
   la condition opératoire (charge et vitesse).
2. FPT (First Predicting Time) par la règle des 3 sigmas sur le RMS de chaque axe :
   alarme si le RMS lissé dépasse moyenne + 3 x écart-type de l'état sain pendant 3 mesures
   consécutives ; l'alarme est annulée s'il repasse sous le seuil pendant 5 minutes (transitoire).
   La règle n'utilise que le passé : elle marche en temps réel.
3. Sélection des features sur les roulements d'apprentissage (après le FPT) selon trois critères
   entre 0 et 1 : monotonie, corrélation avec le temps, pronosticabilité.
4. HI (indicateur de santé) : moyenne des k meilleures features normalisées. On essaie
   k = 1 à 10 et on garde le k qui maximise le score des trois critères sur l'apprentissage
   (« fusion ou sélection » : si k = 1 gagne, le HI est la meilleure feature seule).
   HI = 0 à l'état sain ; il augmente avec la dégradation. Le seuil de défaillance est la valeur
   moyenne du HI à la fin de vie des roulements d'apprentissage.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features import NOMS_PAR_AXE

# toutes les features positives (l'asymétrie, centrée sur 0, n'est pas normalisable par un rapport)
COLONNES = [axe + "_" + nom for axe in ("h", "v") for nom in NOMS_PAR_AXE if nom != "asymetrie"]

DEBUT_SAIN, FIN_SAIN = 30, 150   # fichiers 31 à 150 = état sain de référence
FENETRE_LISSAGE = 6              # moyenne glissante sur 6 mesures (1 minute), uniquement le passé
N_SIGMA = 3
N_CONSECUTIFS = 3
N_ANNULATION = 30                # 30 mesures sous le seuil (5 min) -> fausse alarme annulée
N_FEATURES_MODELES = 8           # les 8 meilleures features servent d'entrées aux modèles ML et deep
K_MAX_HI = 10                    # on teste un HI fait de 1 à 10 features


# ---------------------------------------------------------------- 1. normalisation
def normaliser(features):
    """log(feature / niveau sain), roulement par roulement."""
    resultat = features[["roulement", "index", "temps_s", "rul_s"]].copy()
    for roulement, groupe in features.groupby("roulement"):
        niveau_sain = groupe[COLONNES].iloc[DEBUT_SAIN:FIN_SAIN].mean()
        resultat.loc[groupe.index, COLONNES] = np.log(groupe[COLONNES] / niveau_sain)
    return resultat


def lisser(table, colonnes=COLONNES, fenetre=FENETRE_LISSAGE):
    """Moyenne glissante sur les dernières mesures (pas de valeurs futures)."""
    lisse = table.copy()
    for roulement, groupe in table.groupby("roulement"):
        lisse.loc[groupe.index, colonnes] = groupe[colonnes].rolling(fenetre, min_periods=1).mean()
    return lisse


# ---------------------------------------------------------------- 2. FPT
def seuils_3sigma(groupe):
    """Seuil d'alarme de chaque axe : moyenne + 3 écarts-types du RMS lissé à l'état sain."""
    seuils = {}
    for axe in ("h_rms", "v_rms"):
        sain = groupe[axe].iloc[DEBUT_SAIN:FIN_SAIN]
        seuils[axe] = sain.mean() + N_SIGMA * sain.std()
    return seuils


def detecter_fpt(groupe):
    """Pour un roulement (table lissée), renvoie un tableau : à chaque instant, l'index du fichier
    où l'alarme en cours a commencé, ou 0 si le roulement est jugé sain à cet instant."""
    seuils = seuils_3sigma(groupe)
    alarme = (groupe["h_rms"] > seuils["h_rms"]).to_numpy() | (groupe["v_rms"] > seuils["v_rms"]).to_numpy()
    alarme[:FIN_SAIN] = False          # pas de détection pendant la période de référence

    fpt_courant = np.zeros(len(groupe), dtype=int)
    debut, n_haut, n_bas = 0, 0, 0
    for i in range(len(groupe)):
        if alarme[i]:
            n_haut, n_bas = n_haut + 1, 0
        else:
            n_haut, n_bas = 0, n_bas + 1
        if debut == 0 and n_haut >= N_CONSECUTIFS:
            debut = i - N_CONSECUTIFS + 2          # index du fichier (1, 2, ...) où l'alarme a commencé
        if debut > 0 and n_bas >= N_ANNULATION:
            debut = 0                              # c'était un transitoire : on annule
        fpt_courant[i] = debut
    return fpt_courant


def ajouter_fpt(lisse):
    """Ajoute les colonnes 'fpt_courant' (alarme en cours à chaque instant) et 'fpt' (FPT final)."""
    lisse = lisse.copy()
    for roulement, groupe in lisse.groupby("roulement"):
        courant = detecter_fpt(groupe)
        lisse.loc[groupe.index, "fpt_courant"] = courant
        lisse.loc[groupe.index, "fpt"] = courant[-1]           # FPT retenu = alarme active à la fin
    lisse["fpt_courant"] = lisse["fpt_courant"].astype(int)
    lisse["fpt"] = lisse["fpt"].astype(int)
    return lisse


# ---------------------------------------------------------------- 3. sélection des features
def monotonie(x):
    """|nb de hausses - nb de baisses| / nb de variations : 1 = toujours dans le même sens."""
    d = np.diff(x)
    return abs(np.sum(d > 0) - np.sum(d < 0)) / len(d)


def correlation_temps(x):
    """|corrélation de Spearman| entre la feature et le temps : 1 = tendance parfaite."""
    return abs(spearmanr(x, np.arange(len(x)))[0])


def pronosticabilite(valeurs_initiales, valeurs_finales):
    """Coble & Hines (2009) : 1 si tous les roulements finissent à la même valeur
    (un seuil de défaillance commun est alors facile à fixer)."""
    ecart = np.mean(np.abs(np.array(valeurs_finales) - np.array(valeurs_initiales)))
    return float(np.exp(-np.std(valeurs_finales) / ecart))


def criteres(lisse, roulements, colonne):
    """Les trois critères d'une colonne, calculés sur la phase de dégradation (après le FPT)."""
    mono, corr, debuts, fins = [], [], [], []
    for roulement in roulements:
        groupe = lisse[lisse["roulement"] == roulement]
        phase = groupe[groupe["index"] >= groupe["fpt"].iloc[0]][colonne].to_numpy()
        mono.append(monotonie(phase))
        corr.append(correlation_temps(phase))
        debuts.append(phase[0])
        fins.append(phase[-5:].mean())
    ligne = {"feature": colonne, "monotonie": np.mean(mono), "correlation": np.mean(corr),
             "pronosticabilite": pronosticabilite(debuts, fins)}
    ligne["score"] = (ligne["monotonie"] + ligne["correlation"] + ligne["pronosticabilite"]) / 3
    return ligne


def classer_features(lisse, roulements):
    """Tableau des trois critères pour chaque feature, de la meilleure à la moins bonne."""
    lignes = [criteres(lisse, roulements, colonne) for colonne in COLONNES]
    return pd.DataFrame(lignes).sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 4. indicateur de santé
def calculer_hi(lisse, colonnes):
    """HI = moyenne des features retenues (toutes en log du rapport à l'état sain)."""
    return lisse[colonnes].mean(axis=1)


def choisir_hi(lisse, classement, roulements):
    """Essaie HI = moyenne des k meilleures features (k = 1 à K_MAX_HI) et garde le meilleur score.
    Renvoie les colonnes du HI et le tableau des essais."""
    essais = []
    for k in range(1, K_MAX_HI + 1):
        colonnes = classement["feature"].head(k).tolist()
        essai = lisse[["roulement", "index", "fpt"]].copy()
        essai["hi"] = calculer_hi(lisse, colonnes)
        ligne = criteres(essai, roulements, "hi")
        ligne["k"] = k
        essais.append(ligne)
    essais = pd.DataFrame(essais).drop(columns="feature")
    k_choisi = int(essais.loc[essais["score"].idxmax(), "k"])
    return classement["feature"].head(k_choisi).tolist(), essais

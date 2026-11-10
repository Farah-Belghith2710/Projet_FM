"""Tests simples des briques du projet (lancer :  python -m pytest tests).

On vérifie les fonctions sur des cas dont on connaît la réponse à la main.
"""
import numpy as np
import pandas as pd
import pytest

from src import config, economie, health, reliability, rul_wiener
from src import evaluation as ev
from src.features import extraire


# ---------------------------------------------------------------- métriques du challenge
def test_score_parfait():
    assert ev.score_phm(0) == pytest.approx(1)


def test_score_asymetrique():
    # tardif de 5 % -> 0,5 ; précoce de 20 % -> 0,5 (formule (2) du challenge)
    assert ev.score_phm(-5) == pytest.approx(0.5)
    assert ev.score_phm(20) == pytest.approx(0.5)
    assert ev.score_phm(-10) < ev.score_phm(10)          # le retard est plus pénalisé


def test_erreur_pourcentage():
    assert ev.erreur_pourcentage(100, 80) == pytest.approx(20)     # prédiction trop tôt
    assert ev.erreur_pourcentage(100, 150) == pytest.approx(-50)   # prédiction trop tard


def test_rmse_mae():
    assert ev.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert ev.mae([0, 0], [3, 4]) == pytest.approx(3.5)


def test_bootstrap_contient_la_moyenne():
    valeurs = np.arange(11, dtype=float)
    bas, haut = ev.bootstrap_ic(valeurs)
    assert bas < valeurs.mean() < haut


# ---------------------------------------------------------------- features
def test_rms_d_une_sinusoide():
    t = np.arange(config.N_POINTS) / config.FS
    x = 2 * np.sin(2 * np.pi * 1000 * t)
    assert extraire(x, "")["rms"] == pytest.approx(2 / np.sqrt(2), rel=1e-3)


def test_kurtosis_gaussien_vaut_3():
    x = np.random.default_rng(0).normal(size=200_000)[: config.N_POINTS * 50]
    from scipy.stats import kurtosis
    assert kurtosis(x, fisher=False) == pytest.approx(3, abs=0.05)


def test_frequence_moyenne_d_une_sinusoide():
    t = np.arange(config.N_POINTS) / config.FS
    x = np.sin(2 * np.pi * 2000 * t)
    assert extraire(x, "")["frequence_moyenne"] == pytest.approx(2000, rel=0.01)


def test_bande_ondelettes():
    # une sinusoïde à 2,4 kHz doit mettre son énergie dans la 2e sous-bande (1,6 – 3,2 kHz)
    t = np.arange(config.N_POINTS) / config.FS
    f = extraire(np.sin(2 * np.pi * 2400 * t), "")
    energies = [f[f"wpt_{i}"] for i in range(1, 9)]
    assert int(np.argmax(energies)) == 1


# ---------------------------------------------------------------- FPT
def roulement_synthetique(saut_a=400, transitoire=None, n=600):
    """RMS lissé synthétique : bruit autour de 0, puis une marche à partir de 'saut_a'."""
    rng = np.random.default_rng(1)
    h = rng.normal(0, 0.05, n)
    h[saut_a:] += 1.0
    if transitoire:
        h[transitoire[0]:transitoire[1]] += 1.0
    return pd.DataFrame({"index": np.arange(1, n + 1), "h_rms": h, "v_rms": rng.normal(0, 0.05, n)})


def test_fpt_detecte_le_saut():
    fpt = health.detecter_fpt(roulement_synthetique(saut_a=400))
    assert fpt[-1] == 401                  # index de fichier (commence à 1)
    assert fpt[350] == 0                   # avant le saut : sain


def test_fpt_annule_un_transitoire():
    groupe = roulement_synthetique(saut_a=500, transitoire=(250, 260))
    fpt = health.detecter_fpt(groupe)
    assert fpt[255] > 0                    # alarme pendant le transitoire...
    assert fpt[350] == 0                   # ... annulée ensuite
    assert fpt[-1] == 501


def test_monotonie():
    assert health.monotonie(np.arange(10)) == pytest.approx(1)
    assert health.monotonie(np.array([0, 1, 0, 1, 0])) == pytest.approx(0)


# ---------------------------------------------------------------- fiabilité
def test_weibull_beta_1_est_exponentielle():
    # beta = 1 : loi exponentielle, MTTF = eta et durée de vie résiduelle constante (sans mémoire)
    assert reliability.mttf(1, 100) == pytest.approx(100)
    assert reliability.vie_residuelle_moyenne(50, 1, 100) == pytest.approx(100, rel=1e-4)


def test_b10():
    beta, eta = 2.0, 300.0
    b10 = reliability.quantile_vie(0.10, beta, eta)
    assert reliability.fiabilite(b10, beta, eta) == pytest.approx(0.90)


def test_b10_conditionnel():
    # un roulement sain de 200 min tient encore x min avec 90 % de chances : R(200 + x) / R(200) = 0,9
    beta, eta, age = 1.85, 422.0, 200.0
    x = reliability.vie_residuelle_b10(age, beta, eta)
    rapport = reliability.fiabilite(age + x, beta, eta) / reliability.fiabilite(age, beta, eta)
    assert rapport == pytest.approx(0.90)


def test_l10_condition_1():
    # C = P = 4000 N : 1 million de tours à 1800 tr/min = 555,6 min
    assert reliability.l10_catalogue(1) == pytest.approx(1e6 / 1800)


# ---------------------------------------------------------------- processus de Wiener
PARAMETRES_WIENER = {"mu0": 0.1, "s0": 0.05, "sigma": 0.01, "seuil": 3.0, "seuil_ecart": 0.001}


def test_wiener_derive_constante():
    # HI qui monte de 0,1 par minute : à t = 10 min, HI = 1, il reste (3 - 1) / 0,1 = 20 min
    temps = np.arange(11, dtype=float)
    mediane, bas, haut = rul_wiener.predire(temps, 0.1 * temps, PARAMETRES_WIENER)
    assert mediane[-1] == pytest.approx(20, rel=0.02)
    assert bas[-1] <= mediane[-1] <= haut[-1]


def test_wiener_seuil_deja_franchi():
    temps = np.arange(5, dtype=float)
    mediane, _, _ = rul_wiener.predire(temps, np.full(5, 3.5), PARAMETRES_WIENER)
    assert mediane[-1] == 0


# ---------------------------------------------------------------- économie (décision de maintenance)
def test_cout_tout_correctif():
    # 2 roulements qui cassent à 60 et 120 min : 2 pannes x 5 u en 3 h de marche -> 3,33 u/h
    bilan = economie.evaluer([None, None], [60, 120], cp=1, cc=5)
    assert bilan["cout_par_heure_u"] == pytest.approx(10 / 3, abs=1e-3)
    assert bilan["pannes"] == 2


def test_remplacement_avant_la_panne():
    # remplacé à 50 min alors qu'il aurait cassé à 60 min : coût planifié et 10 min de vie gaspillée
    bilan = economie.evaluer([50], [60], cp=1, cc=5)
    assert bilan["pannes"] == 0
    assert bilan["vie_perdue_moyenne_min"] == 10


def test_age_optimal_avec_usure():
    # usure (beta > 1) et panne 5 fois plus chère : remplacer à T* coûte moins cher que tout correctif
    t_etoile, cout = economie.age_optimal(2.0, 400, 1, 5)
    assert 0 < t_etoile < 400
    assert cout < 5 / reliability.mttf(2.0, 400) * 60


def test_declenchement_sous_le_seuil():
    assert economie.instant_declenchement(np.array([0, 1, 2, 3]), [30, 20, 9, 5], 10) == 2
    assert economie.instant_declenchement(np.array([0, 1]), [30, 20], 10) is None

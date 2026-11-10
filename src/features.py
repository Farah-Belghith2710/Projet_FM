"""Sprint 2 — Extraction des caractéristiques (features) d'un signal de 0,1 s.

Trois familles, calculées pour l'axe horizontal (h_) et l'axe vertical (v_) :
- temporelles : RMS, pic, crête-à-crête, kurtosis, asymétrie, facteurs de crête, de forme, d'impulsion ;
- fréquentielles : fréquence moyenne (centre de gravité du spectre) et entropie spectrale ;
- temps-fréquence : énergie des 8 sous-bandes d'une décomposition en paquets d'ondelettes
  (niveau 3, ondelette db4). Chaque sous-bande couvre 1,6 kHz (0–1,6 kHz, 1,6–3,2 kHz, ...).
"""
import numpy as np
import pywt
from scipy.stats import kurtosis, skew

from src import config

FREQUENCES = np.fft.rfftfreq(config.N_POINTS, d=1 / config.FS)

NOMS_TEMPORELS = ["rms", "pic", "crete_crete", "kurtosis", "asymetrie",
                  "facteur_crete", "facteur_forme", "facteur_impulsion"]
NOMS_FREQUENTIELS = ["frequence_moyenne", "entropie_spectrale"]
NOMS_ONDELETTES = [f"wpt_{i}" for i in range(1, 9)]
NOMS_PAR_AXE = NOMS_TEMPORELS + NOMS_FREQUENTIELS + NOMS_ONDELETTES


def features_temporelles(x):
    rms = np.sqrt(np.mean(x ** 2))
    pic = np.max(np.abs(x))
    moyenne_abs = np.mean(np.abs(x))
    return {
        "rms": rms,
        "pic": pic,
        "crete_crete": np.max(x) - np.min(x),
        "kurtosis": kurtosis(x, fisher=False),   # vaut 3 pour un signal gaussien
        "asymetrie": skew(x),
        "facteur_crete": pic / rms,
        "facteur_forme": rms / moyenne_abs,
        "facteur_impulsion": pic / moyenne_abs,
    }


def features_frequentielles(x):
    puissance = np.abs(np.fft.rfft(x)) ** 2
    p = puissance / puissance.sum()                         # spectre normalisé (somme = 1)
    entropie = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))  # entre 0 et 1
    return {
        "frequence_moyenne": np.sum(FREQUENCES * p),
        "entropie_spectrale": entropie,
    }


def features_ondelettes(x):
    paquets = pywt.WaveletPacket(data=x, wavelet="db4", mode="symmetric", maxlevel=3)
    noeuds = paquets.get_level(3, order="freq")             # 8 sous-bandes, de la plus basse à la plus haute
    return {f"wpt_{i + 1}": np.sum(noeud.data ** 2) for i, noeud in enumerate(noeuds)}


def extraire(x, prefixe):
    """Toutes les features d'un signal filtré, avec un préfixe 'h_' ou 'v_'."""
    resultat = {}
    resultat.update(features_temporelles(x))
    resultat.update(features_frequentielles(x))
    resultat.update(features_ondelettes(x))
    return {prefixe + nom: valeur for nom, valeur in resultat.items()}

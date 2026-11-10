"""Sprint 1 — Prétraitement des signaux vibratoires.

Deux opérations simples, appliquées à chaque fichier de 0,1 s :
1. suppression de la dérive (tendance linéaire) ;
2. filtre passe-bande 10 Hz – 10 kHz : c'est la plage utile de l'accéléromètre DYTRAN 3035B
   (0,5 Hz à 10 kHz d'après sa fiche technique). On coupe les très basses fréquences
   (dérives, bruit de la structure) et le bruit au-dessus de 10 kHz.
"""
from scipy.signal import butter, detrend, sosfiltfilt

from src import config

BANDE_HZ = (10, 10000)

# Le filtre est calculé une seule fois (Butterworth d'ordre 4)
FILTRE = butter(4, BANDE_HZ, btype="bandpass", fs=config.FS, output="sos")


def nettoyer(signal):
    """Signal brut -> signal sans dérive et filtré (filtrage aller-retour : pas de déphasage)."""
    signal = detrend(signal)
    return sosfiltfilt(FILTRE, signal)

"""Sprint 3 — Modèle stochastique : processus de Wiener avec mise à jour bayésienne.

Après le FPT, le HI évolue comme   X(t) = X(t_FPT) + mu.(t - t_FPT) + sigma.B(t - t_FPT)
(B = mouvement brownien). La RUL est le temps mis par le HI pour atteindre le seuil de défaillance D.

- La dérive mu varie beaucoup d'un roulement à l'autre : on lui donne une loi a priori N(mu0, s0²)
  apprise sur les 6 roulements d'apprentissage.
- À chaque nouvelle mesure, on combine cet a priori avec la montée observée du HI
  (mise à jour bayésienne conjuguée) : au début on fait confiance à la flotte,
  puis de plus en plus au roulement lui-même.
- Le seuil de défaillance D est lui aussi incertain : les roulements d'apprentissage ne
  défaillent pas tous au même niveau de HI. On le tire dans N(D moyen, écart-type observé²).
- Pour une dérive mu et un seuil D fixés, le temps de premier passage suit une loi inverse
  gaussienne de moyenne (D - X)/mu et de forme (D - X)²/sigma². On tire mu et D, puis la RUL :
  on obtient la médiane et un intervalle de prédiction à 90 % (quantification de l'incertitude).
"""
import numpy as np

RUL_MAX_MIN = 600     # plafond de 10 h pour les prédictions (évite les valeurs infinies)


def estimer_parametres(lisse, roulements, colonne="hi"):
    """Apprend mu0, s0 (loi a priori de la dérive), sigma (bruit) et le seuil D sur l'apprentissage."""
    derives, carres, finaux = [], [], []
    for roulement in roulements:
        groupe = lisse[lisse["roulement"] == roulement]
        phase = groupe[groupe["index"] >= groupe["fpt"].iloc[0]]
        x = phase[colonne].to_numpy()
        t = phase["temps_s"].to_numpy() / 60                 # minutes

        mu = (x[-1] - x[0]) / (t[-1] - t[0])                 # dérive moyenne de ce roulement
        derives.append(mu)

        # incréments sur 1 minute (6 mesures) pour estimer le bruit sigma
        increments = (x[6:] - x[:-6])[::6]
        carres.extend((increments - mu * 1.0) ** 2 / 1.0)
        finaux.append(x[-5:].mean())                         # HI à la défaillance

    return {
        "mu0": float(np.mean(derives)), "s0": float(np.std(derives)),
        "sigma": float(np.sqrt(np.mean(carres))),
        "seuil": float(np.mean(finaux)), "seuil_ecart": float(np.std(finaux)),
        "derives": [float(d) for d in derives], "hi_finaux": [float(f) for f in finaux],
    }


def predire(temps_min, hi, parametres, n_tirages=2000, graine=0):
    """RUL (minutes) à chaque instant depuis le FPT : médiane, borne 5 % et borne 95 %.
    temps_min et hi commencent au FPT ; à l'instant k, seules les mesures 0..k sont utilisées."""
    rng = np.random.default_rng(graine)
    mu0, s0, sigma = parametres["mu0"], parametres["s0"], parametres["sigma"]

    mediane, bas, haut = [], [], []
    for k in range(len(hi)):
        duree = temps_min[k] - temps_min[0]
        montee = hi[k] - hi[0]

        # loi a posteriori de la dérive (normale) : a priori + observations du roulement
        precision = 1 / s0 ** 2 + duree / sigma ** 2
        mu_post = (mu0 / s0 ** 2 + montee / sigma ** 2) / precision
        ecart_post = 1 / np.sqrt(precision)

        # tirages de la dérive (positive : la dégradation ne recule pas) et du seuil
        mus = rng.normal(mu_post, ecart_post, n_tirages)
        seuils = rng.normal(parametres["seuil"], parametres["seuil_ecart"], n_tirages)
        garder = mus > 1e-4
        mus, seuils = mus[garder], seuils[garder]
        if len(mus) == 0:
            mediane.append(RUL_MAX_MIN), bas.append(RUL_MAX_MIN), haut.append(RUL_MAX_MIN)
            continue

        reste = seuils - hi[k]
        ruls = np.zeros(len(mus))            # reste <= 0 : ce seuil est déjà franchi, RUL = 0
        actif = reste > 0
        ruls[actif] = rng.wald(reste[actif] / mus[actif], reste[actif] ** 2 / sigma ** 2)  # inverse gaussienne
        ruls = np.minimum(ruls, RUL_MAX_MIN)
        p05, p50, p95 = np.percentile(ruls, [5, 50, 95])
        mediane.append(p50), bas.append(p05), haut.append(p95)

    return np.array(mediane), np.array(bas), np.array(haut)

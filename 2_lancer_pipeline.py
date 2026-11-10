"""ÉTAPE 2 (Sprints 2, 3 et 4) — table de features -> tous les résultats (dossier results/).

    python 2_lancer_pipeline.py            tout recalculer (7 à 10 minutes sur un PC portable)
    python 2_lancer_pipeline.py --rapide   sans recherche d'hyperparamètres pour le deep learning (1 à 3 min)

Étapes :
  1. santé      : normalisation, lissage, FPT (3 sigmas), classement des features, HI     (Sprint 2)
  2. fiabilité  : Kaplan-Meier, Weibull, MTTF, L10, durée de vie résiduelle (B10, MRL)     (Sprint 1)
  3. modèles    : Wiener, SVR, GPR, Random Forest, LSTM, CNN 1D                           (Sprint 3)
  4. évaluation : RMSE, MAE, score PHM, intervalles bootstrap, score officiel du challenge (Sprint 4)

Si TensorFlow n'est pas installé, les deux modèles de deep learning sont simplement ignorés.
"""
import argparse
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # TensorFlow : n'afficher que les erreurs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # TensorFlow : calculs plus reproductibles d'un PC à l'autre
os.environ["PYTHONWARNINGS"] = "ignore"       # avertissements sans conséquence (convergence du GPR, ...)
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)   # afficher la progression ligne par ligne

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, health, reliability, rul_ml, rul_wiener  # noqa: E402
from src import evaluation as ev  # noqa: E402

FAMILLES = {
    "Fiabilité (B10)": "Fiabilité (sans capteur)",
    "Wiener": "Stochastique",
    "SVR": "ML classique",
    "GPR": "ML classique",
    "Random Forest": "ML classique",
    "LSTM": "Deep learning",
    "CNN 1D": "Deep learning",
}
POINTS_EVALUATION = [0.25, 0.50, 0.75]    # instants d'évaluation : 25 %, 50 % et 75 % de la phase de dégradation


# ---------------------------------------------------------------- petits outils
def ecrire_json(nom, objet):
    with open(config.DOSSIER_RESULTATS / nom, "w", encoding="utf-8") as f:
        json.dump(objet, f, ensure_ascii=False, indent=1)
    print(f"   -> results/{nom}")


def arrondir(valeurs, n=2):
    return [round(float(v), n) for v in valeurs]


def episodes_alarme(fpt_courant):
    """Liste des alarmes [début, fin] (index de fichiers) ; une alarme qui s'arrête avant la fin a été annulée."""
    episodes, debut = [], 0
    for i, valeur in enumerate(fpt_courant):
        if valeur > 0 and debut == 0:
            debut = valeur
        if valeur == 0 and debut > 0:
            episodes.append([int(debut), int(i)])
            debut = 0
    if debut > 0:
        episodes.append([int(debut), len(fpt_courant)])
    return episodes


# ---------------------------------------------------------------- 1. santé
def etape_sante(features):
    print("1. Santé : normalisation, FPT, sélection des features, indicateur de santé")
    lisse = health.ajouter_fpt(health.lisser(health.normaliser(features)))
    classement = health.classer_features(lisse, config.APPRENTISSAGE)
    selection = classement["feature"].head(health.N_FEATURES_MODELES).tolist()   # entrées des modèles
    colonnes_hi, essais_hi = health.choisir_hi(lisse, classement, config.APPRENTISSAGE)
    lisse["hi"] = health.calculer_hi(lisse, colonnes_hi)
    criteres_hi = health.criteres(lisse, config.APPRENTISSAGE, "hi")
    print(f"   HI = moyenne de {len(colonnes_hi)} feature(s) : {', '.join(colonnes_hi)}")

    par_roulement = {}
    for roulement, groupe in lisse.groupby("roulement"):
        seuils = health.seuils_3sigma(groupe)
        fpt = int(groupe["fpt"].iloc[0])
        par_roulement[roulement] = {
            "fpt_index": fpt,
            "fpt_min": round(fpt * config.PAS_TEMPS / 60, 2),
            "phase_degradation_min": round((groupe["index"].iloc[-1] - fpt) * config.PAS_TEMPS / 60, 2),
            "seuil_h": round(float(seuils["h_rms"]), 4),
            "seuil_v": round(float(seuils["v_rms"]), 4),
            "episodes": episodes_alarme(groupe["fpt_courant"].to_numpy()),
        }
    sante = {
        "parametres": {"etat_sain_fichiers": [health.DEBUT_SAIN + 1, health.FIN_SAIN],
                       "fenetre_lissage": health.FENETRE_LISSAGE, "n_sigma": health.N_SIGMA,
                       "n_consecutifs": health.N_CONSECUTIFS, "n_annulation": health.N_ANNULATION},
        "classement": classement.round(4).to_dict(orient="records"),
        "selection": selection,
        "colonnes_hi": colonnes_hi,
        "choix_hi": essais_hi.round(4).to_dict(orient="records"),
        "criteres_hi": {k: (round(v, 4) if k != "feature" else v) for k, v in criteres_hi.items()},
        "roulements": par_roulement,
    }
    return lisse, selection, sante


# ---------------------------------------------------------------- 2. fiabilité
def etape_fiabilite(roulements):
    print("2. Fiabilité : Kaplan-Meier, Weibull, L10, durée de vie résiduelle (B10, MRL)")
    durees, evenements = reliability.donnees_survie(roulements)
    beta, eta = reliability.ajuster_weibull(durees, evenements)
    t = np.linspace(1, 700, 300)
    vies = roulements["duree_vie_s"] / 60

    fiabilite = {
        "donnees": [{"roulement": r, "duree_min": round(float(d), 1), "evenement": int(e)}
                    for r, d, e in zip(roulements["roulement"], durees, evenements)],
        "kaplan_meier": {k: arrondir(v, 4) for k, v in reliability.kaplan_meier(durees, evenements).items()},
        "weibull": {"beta": round(beta, 3), "eta_min": round(eta, 1),
                    "mttf_min": round(reliability.mttf(beta, eta), 1),
                    "b10_min": round(reliability.quantile_vie(0.10, beta, eta), 1),
                    "b50_min": round(reliability.quantile_vie(0.50, beta, eta), 1)},
        "courbes": {"t_min": arrondir(t, 1),
                    "fiabilite": arrondir(reliability.fiabilite(t, beta, eta), 4),
                    "taux_par_heure": arrondir(60 * reliability.taux_defaillance(t, beta, eta), 5),
                    "mrl_min": arrondir([reliability.vie_residuelle_moyenne(x, beta, eta) for x in t], 1),
                    "b10_residuel_min": arrondir(reliability.vie_residuelle_b10(t, beta, eta), 1)},
        "l10": [{"condition": c,
                 "l10_min": round(reliability.l10_catalogue(c), 1),
                 "vies_observees_min": arrondir(vies[roulements["condition"] == c], 1),
                 "roulements": roulements.loc[roulements["condition"] == c, "roulement"].tolist()}
                for c in config.CONDITIONS],
    }
    return beta, eta, fiabilite


# ---------------------------------------------------------------- 3. modèles
def etape_modeles(lisse, selection, rapide):
    print("3. Modèles de RUL")
    lisse = rul_ml.ajouter_depuis_fpt(lisse)
    entrees = selection + ["depuis_fpt_min"]
    modeles, infos = {}, {}

    debut = time.time()
    parametres_wiener = rul_wiener.estimer_parametres(lisse, config.APPRENTISSAGE)
    infos["Wiener"] = {"parametres": {k: (round(v, 4) if isinstance(v, float) else arrondir(v, 4))
                                      for k, v in parametres_wiener.items()},
                       "duree_s": round(time.time() - debut, 2)}
    print(f"   Wiener          seuil D = {parametres_wiener['seuil']:.2f}, dérive a priori = "
          f"{parametres_wiener['mu0']:.3f} ± {parametres_wiener['s0']:.3f} /min")

    X, y, groupes = rul_ml.jeu_apprentissage(lisse, config.APPRENTISSAGE, entrees)
    for nom in ["SVR", "GPR", "Random Forest"]:
        modeles[nom], infos[nom] = rul_ml.entrainer(nom, X, y, groupes)
        print(f"   {nom:15s} {infos[nom]['meilleurs_parametres']}  ({infos[nom]['duree_s']} s)")

    try:
        from src import rul_deep
    except ImportError:
        rul_deep = None
        print("   TensorFlow n'est pas installé : LSTM et CNN 1D ignorés")
    if rul_deep is not None:
        for nom in ["LSTM", "CNN 1D"]:
            modeles[nom], infos[nom] = rul_deep.entrainer(nom, lisse, config.APPRENTISSAGE, selection,
                                                          recherche=not rapide)
            print(f"   {nom:15s} {infos[nom]['meilleurs_parametres']}  ({infos[nom]['duree_s']} s)")
    return lisse, entrees, parametres_wiener, modeles, infos, rul_deep


def predictions_roulement(groupe, entrees, selection, beta, eta, parametres_wiener, modeles, rul_deep):
    """RUL prédite par chaque modèle à chaque instant après le FPT (en minutes)."""
    post = groupe[groupe["index"] >= groupe["fpt"].iloc[0]]
    temps = post["temps_s"].to_numpy() / 60
    predictions = {"Fiabilité (B10)": reliability.vie_residuelle_b10(temps, beta, eta)}
    mediane, bas, haut = rul_wiener.predire(temps, post["hi"].to_numpy(), parametres_wiener)
    predictions["Wiener"] = mediane
    for nom, modele in modeles.items():
        if nom in ("LSTM", "CNN 1D"):
            predictions[nom] = rul_deep.predire(modele, groupe, selection)
        else:
            predictions[nom] = rul_ml.predire(modele, post[entrees].to_numpy())
    return post, predictions, bas, haut


# ---------------------------------------------------------------- 4. évaluation
def etape_evaluation(lisse, entrees, selection, beta, eta, parametres_wiener, modeles, rul_deep):
    print("4. Évaluation sur les 11 roulements de test")
    noms = ["Fiabilité (B10)", "Wiener"] + list(modeles)
    courbes, troncature = {}, []
    a_mrl, er_mrl = [], []          # référence : même protocole avec la durée de vie résiduelle MOYENNE
    rmse_b = {n: [] for n in noms}
    mae_b = {n: [] for n in noms}
    score_b = {n: [] for n in noms}
    a_tronc = {n: [] for n in noms}
    er_tronc = {n: [] for n in noms}
    couverture = []

    for roulement in config.TEST:
        groupe = lisse[lisse["roulement"] == roulement].reset_index(drop=True)
        post, predictions, bas, haut = predictions_roulement(groupe, entrees, selection, beta, eta,
                                                             parametres_wiener, modeles, rul_deep)
        vrai = post["rul_s"].to_numpy() / 60
        positions = [int(round(q * (len(vrai) - 1))) for q in POINTS_EVALUATION]

        for nom in noms:
            p = predictions[nom]
            rmse_b[nom].append(ev.rmse(vrai, p))
            mae_b[nom].append(ev.mae(vrai, p))
            erreurs = [ev.erreur_pourcentage(vrai[k], p[k]) for k in positions]
            score_b[nom].append(float(np.mean(ev.score_phm(erreurs))))
        couverture.append(float(np.mean((vrai >= bas) & (vrai <= haut))))

        # protocole officiel : une seule prédiction, à l'instant de troncature
        fpt = int(groupe["fpt"].iloc[0])
        k = config.TEST[roulement]["troncature"]
        rul_officielle = config.TEST[roulement]["rul_reelle"] / 60
        etat_a_k = int(groupe["fpt_courant"].iloc[k - 1])
        degrade = etat_a_k > 0 and etat_a_k == fpt
        age = k * config.PAS_TEMPS / 60
        ligne = {"roulement": roulement, "etat": "dégradé" if degrade else "sain",
                 "age_min": round(age, 2), "rul_officielle_min": round(rul_officielle, 2)}
        er = float(ev.erreur_pourcentage(rul_officielle, reliability.vie_residuelle_moyenne(age, beta, eta)))
        er_mrl.append(er)
        a_mrl.append(float(ev.score_phm(er)))
        for nom in noms:
            if degrade:
                p = float(predictions[nom][k - fpt])
            else:                          # aucun défaut détecté : seule la fiabilité peut répondre
                p = float(reliability.vie_residuelle_b10(age, beta, eta))
            er = float(ev.erreur_pourcentage(rul_officielle, p))
            a_tronc[nom].append(float(ev.score_phm(er)))
            er_tronc[nom].append(er)
            ligne[nom] = {"rul_predite_min": round(p, 2), "erreur_pct": round(er, 1),
                          "score": round(float(ev.score_phm(er)), 4)}
        troncature.append(ligne)

        courbes[roulement] = {
            "fpt_min": round(fpt * config.PAS_TEMPS / 60, 2),
            "troncature_min": round(k * config.PAS_TEMPS / 60, 2),
            "temps_min": arrondir(post["temps_s"] / 60, 2),
            "vrai_min": arrondir(vrai, 2),
            "predictions": {nom: arrondir(predictions[nom], 2) for nom in noms},
            "wiener_bas": arrondir(bas, 2), "wiener_haut": arrondir(haut, 2),
            "points_evaluation": positions,
        }

    tableau = []
    for nom in noms:
        tableau.append({
            "modele": nom, "famille": FAMILLES[nom],
            "rmse_min": round(float(np.mean(rmse_b[nom])), 2),
            "mae_min": round(float(np.mean(mae_b[nom])), 2),
            "score_phase": round(float(np.mean(score_b[nom])), 4),
            "score_phase_ic95": arrondir(ev.bootstrap_ic(score_b[nom]), 4),
            "rmse_ic95": arrondir(ev.bootstrap_ic(rmse_b[nom]), 2),
            "score_challenge": round(float(np.mean(a_tronc[nom])), 4),
            "score_challenge_ic95": arrondir(ev.bootstrap_ic(a_tronc[nom]), 4),
            "erreur_abs_moyenne_pct": round(float(np.mean(np.abs(er_tronc[nom]))), 1),
        })
    par_roulement = {nom: {"rmse_min": arrondir(rmse_b[nom]), "mae_min": arrondir(mae_b[nom]),
                           "score_phase": arrondir(score_b[nom], 4)} for nom in noms}
    return {"modeles": noms, "metriques": tableau, "par_roulement": par_roulement,
            "roulements": list(config.TEST), "troncature": troncature, "courbes": courbes,
            "couverture_wiener_90": {"par_roulement": arrondir(couverture, 3),
                                     "moyenne": round(float(np.mean(couverture)), 3)},
            "reference_mrl": {"score_challenge": round(float(np.mean(a_mrl)), 4),
                              "erreur_abs_moyenne_pct": round(float(np.mean(np.abs(er_mrl))), 1),
                              "explication": "fiabilité seule, durée de vie résiduelle moyenne (MRL) au lieu du B10"},
            "points_evaluation": POINTS_EVALUATION}


# ---------------------------------------------------------------- export de la série temporelle
def exporter_series(features, lisse, selection):
    """Une ligne par fichier : quelques features brutes, les features normalisées retenues et le HI."""
    series = pd.DataFrame({
        "roulement": features["roulement"],
        "ensemble": np.where(features["roulement"].isin(config.APPRENTISSAGE), "apprentissage", "test"),
        "index": features["index"],
        "temps_min": (features["temps_s"] / 60).round(3),
        "rul_min": (features["rul_s"] / 60).round(3),
    })
    for colonne in ["h_rms", "v_rms", "h_kurtosis", "v_kurtosis", "h_max", "h_min", "v_max", "v_min"]:
        series[colonne + "_brut"] = features[colonne].round(4)
    for colonne in ["h_rms", "v_rms"] + [c for c in selection if c not in ("h_rms", "v_rms")]:
        series[colonne + "_norm"] = lisse[colonne].round(4)
    series["hi"] = lisse["hi"].round(4)
    series["fpt_courant"] = lisse["fpt_courant"]
    series["fpt"] = lisse["fpt"]
    series.to_csv(config.DOSSIER_RESULTATS / "series.csv", index=False)
    print("   -> results/series.csv")


def version_de(*paquets):
    """Version installée du premier nom de paquet trouvé (None s'il est absent)."""
    for paquet in paquets:
        try:
            return version(paquet)
        except PackageNotFoundError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapide", action="store_true", help="sans recherche d'hyperparamètres (deep learning)")
    args = parser.parse_args()

    debut = time.time()
    config.DOSSIER_RESULTATS.mkdir(exist_ok=True)
    features = pd.read_csv(config.FICHIER_FEATURES)
    roulements = pd.read_csv(config.DOSSIER_TRAITE / "roulements.csv")

    lisse, selection, sante = etape_sante(features)
    beta, eta, fiabilite = etape_fiabilite(roulements)
    lisse, entrees, parametres_wiener, modeles, infos, rul_deep = etape_modeles(lisse, selection, args.rapide)
    rul = etape_evaluation(lisse, entrees, selection, beta, eta, parametres_wiener, modeles, rul_deep)
    rul["infos_modeles"] = infos
    rul["entrees"] = entrees

    # tableau des roulements enrichi (FPT, phase de dégradation)
    donnees = roulements.copy()
    donnees["fpt_min"] = donnees["roulement"].map(lambda r: sante["roulements"][r]["fpt_min"])
    donnees["phase_degradation_min"] = donnees["roulement"].map(lambda r: sante["roulements"][r]["phase_degradation_min"])
    donnees["duree_vie_min"] = (donnees["duree_vie_s"] / 60).round(2)

    print("5. Export")
    ecrire_json("donnees.json", {"roulements": json.loads(donnees.to_json(orient="records")),
                                 "conditions": config.CONDITIONS})
    ecrire_json("sante.json", sante)
    ecrire_json("fiabilite.json", fiabilite)
    ecrire_json("rul.json", rul)
    exporter_series(features, lisse, selection)
    ecrire_json("run_info.json", {
        "date": datetime.now().isoformat(timespec="seconds"),
        "duree_s": round(time.time() - debut, 1),
        "mode": "rapide" if args.rapide else "complet",
        "python": platform.python_version(),
        "systeme": f"{platform.system()} {platform.release()}",
        "bibliotheques": {p: version_de(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "lifelines",
                                                      "PyWavelets", "keras", "streamlit", "plotly"]}
                         | {"tensorflow": version_de("tensorflow", "tensorflow-cpu")},
        "graine": config.GRAINE,
    })

    tableau = pd.DataFrame(rul["metriques"]).set_index("modele")
    print("\nRésultats (11 roulements de test) :")
    print(tableau[["famille", "rmse_min", "mae_min", "score_phase", "score_challenge"]].to_string())
    print(f"\nTerminé en {time.time() - debut:.0f} s")


if __name__ == "__main__":
    main()

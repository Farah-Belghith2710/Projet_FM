"""ÉTAPE 4 — Analyses complémentaires pour le rapport -> results/complements.json

    python 4_analyses_complementaires.py        (environ 5 minutes, sans TensorFlow)

  1. Vraisemblance   : 4 lois de survie comparées, test du rapport de vraisemblance, surface de vraisemblance
  2. B10 et B50      : intervalles de confiance par bootstrap (on retire les roulements au hasard)
  3. Covariables     : Weibull à temps accéléré (charge, vitesse) et modèles ML avec charge et vitesse en entrée
  4. Données simulées: chaque méthode retrouve-t-elle la vérité quand on la connaît ?
  5. Sensibilité     : seuil du FPT (2 à 5 sigmas), variantes du Wiener, grille du SVR
  6. Explicabilité   : importance de chaque entrée des modèles ML (permutation)
  7. Tests statistiques : Wilcoxon et Friedman sur les erreurs par roulement
  8. Économie        : coût par heure de chaque stratégie de maintenance, âge optimal, seuil de décision

Lit features.csv, roulements.csv et les résultats de l'étape 2 (results/*.json), qui restent la référence.
"""
import importlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from lifelines import ExponentialFitter, LogLogisticFitter, LogNormalFitter, WeibullAFTFitter, WeibullFitter
from scipy import stats
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from src import config, economie, health, reliability, rul_ml, rul_wiener
from src import evaluation as ev

pipeline = importlib.import_module("2_lancer_pipeline")    # on réutilise les étapes du pipeline principal
warnings.filterwarnings("ignore")
arrondir = pipeline.arrondir


def lire_resultat(nom):
    with open(config.DOSSIER_RESULTATS / nom, encoding="utf-8") as f:
        return json.load(f)


def resume(valeurs, n=2):
    """Moyenne, médiane et intervalle à 90 % (5 % - 95 %) d'une liste de valeurs."""
    valeurs = np.asarray(valeurs, float)
    return {"moyenne": round(float(valeurs.mean()), n), "mediane": round(float(np.median(valeurs)), n),
            "p05": round(float(np.percentile(valeurs, 5)), n), "p95": round(float(np.percentile(valeurs, 95)), n)}


# ---------------------------------------------------------------- 1. vraisemblance
def vraisemblance(durees, evenements):
    print("1. Vraisemblance : comparaison de 4 lois de survie")
    lois = {"Exponentielle": ExponentialFitter(), "Weibull": WeibullFitter(),
            "Log-normale": LogNormalFitter(), "Log-logistique": LogLogisticFitter()}
    t = np.linspace(1, 700, 140)
    tableau, courbes = [], {"t_min": arrondir(t, 1)}
    for nom, loi in lois.items():
        loi.fit(durees, event_observed=evenements)
        tableau.append({"loi": nom, "n_parametres": len(loi.params_),
                        "log_vraisemblance": round(float(loi.log_likelihood_), 2), "aic": round(float(loi.AIC_), 2),
                        "parametres": {k: round(float(v), 3) for k, v in loi.params_.items()}})
        courbes[nom] = arrondir(loi.survival_function_at_times(t).to_numpy(), 4)

    # test du rapport de vraisemblance : Weibull contre exponentielle (= Weibull avec beta = 1)
    lr = 2 * (lois["Weibull"].log_likelihood_ - lois["Exponentielle"].log_likelihood_)
    p_valeur = float(stats.chi2.sf(lr, df=1))

    # surface de log-vraisemblance de Weibull (avec censure) sur une grille (beta, eta)
    betas, etas = np.linspace(0.8, 4.0, 65), np.linspace(220, 900, 69)
    B, E = np.meshgrid(betas, etas)
    log_v = np.zeros_like(B)
    for duree, evenement in zip(durees, evenements):
        log_v += evenement * (np.log(B) - np.log(E) + (B - 1) * np.log(duree / E)) - (duree / E) ** B
    meilleure = min(tableau, key=lambda ligne: ligne["aic"])["loi"]
    print(f"   loi au plus petit AIC : {meilleure} ; test Weibull vs exponentielle : p = {p_valeur:.3f}")
    return {
        "lois": tableau, "meilleure_loi_aic": meilleure, "courbes": courbes,
        "test_rapport_vraisemblance": {"statistique": round(float(lr), 3), "p_valeur": round(p_valeur, 4),
                                       "conclusion": "beta différent de 1" if p_valeur < 0.05 else "beta = 1 non rejeté"},
        "surface": {"beta": arrondir(betas, 3), "eta_min": arrondir(etas, 1), "log_vraisemblance": [arrondir(l, 2) for l in log_v],
                    "seuil_region_95": round(float(log_v.max() - stats.chi2.ppf(0.95, 2) / 2), 2)},
    }


# ---------------------------------------------------------------- 2. B10 et B50 avec intervalles de confiance
def intervalles_bootstrap(durees, evenements, n_tirages=1000, graine=0):
    print("2. B10 et B50 : bootstrap sur les 17 roulements")
    rng = np.random.default_rng(graine)
    tirages = {"beta": [], "eta_min": [], "mttf_min": [], "b10_min": [], "b50_min": []}
    for _ in range(n_tirages):
        choix = rng.integers(0, len(durees), len(durees))        # 17 roulements tirés avec remise
        if evenements[choix].sum() < 2:                          # il faut au moins 2 défaillances
            continue
        beta, eta = reliability.ajuster_weibull(durees[choix], evenements[choix])
        tirages["beta"].append(beta)
        tirages["eta_min"].append(eta)
        tirages["mttf_min"].append(reliability.mttf(beta, eta))
        tirages["b10_min"].append(reliability.quantile_vie(0.10, beta, eta))
        tirages["b50_min"].append(reliability.quantile_vie(0.50, beta, eta))
    ic = {nom: [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
          for nom, v in tirages.items()}
    print(f"   B10 : IC 95 % = {ic['b10_min']} min ; B50 : {ic['b50_min']} min")
    return {"n_tirages_valides": len(tirages["beta"]), "ic95": ic}


# ---------------------------------------------------------------- 3. covariables
def covariables_fiabilite(roulements, durees, evenements):
    print("3a. Covariables : Weibull à temps accéléré (AFT)")
    conditions = roulements["condition"].to_numpy()
    table = pd.DataFrame({
        "duree": durees, "evenement": evenements,
        "charge_kN": [config.CONDITIONS[c]["charge_N"] / 1000 for c in conditions],
        "vitesse_100tr": [config.CONDITIONS[c]["vitesse_rpm"] / 100 for c in conditions],
        "log_l10": [np.log(reliability.l10_catalogue(c)) for c in conditions],
    })
    sans = WeibullFitter().fit(durees, event_observed=evenements)
    modeles = [{"covariable": "aucune", "aic": round(float(sans.AIC_), 2)}]
    t = np.linspace(1, 700, 140)
    courbes = {"t_min": arrondir(t, 1)}
    for nom, colonne in [("charge (kN)", "charge_kN"), ("vitesse (par 100 tr/min)", "vitesse_100tr"),
                         ("log L10 (ISO 281)", "log_l10")]:
        aft = WeibullAFTFitter().fit(table[["duree", "evenement", colonne]], "duree", "evenement")
        coef = float(aft.params_.loc[("lambda_", colonne)])
        ligne = {"covariable": nom, "coefficient": round(coef, 3),
                 "facteur_duree_de_vie": round(float(np.exp(coef)), 3),
                 "p_valeur": round(float(aft.summary.loc[("lambda_", colonne), "p"]), 3),
                 "aic": round(float(aft.AIC_), 2)}
        if colonne == "charge_kN":                               # B10 et courbe R(t) par condition
            par_condition = table.drop_duplicates(colonne)[[colonne]].sort_values(colonne)
            b10 = aft.predict_percentile(par_condition, p=0.90).to_numpy()
            survie = aft.predict_survival_function(par_condition, times=t)
            ligne["b10_par_condition_min"] = {str(c): round(float(b), 1) for c, b in zip((1, 2, 3), b10)}
            for c, colonne_survie in zip((1, 2, 3), survie.columns):
                courbes[f"condition {c}"] = arrondir(survie[colonne_survie].to_numpy(), 4)
        modeles.append(ligne)
        print(f"   {nom:26s} facteur {ligne['facteur_duree_de_vie']:.3f}  p = {ligne['p_valeur']:.3f}  "
              f"AIC = {ligne['aic']:.1f} (sans covariable : {modeles[0]['aic']:.1f})")
    return {"modeles": modeles, "courbes_charge": courbes}


def entrainer_ml(lisse, entrees):
    X, y, groupes = rul_ml.jeu_apprentissage(lisse, config.APPRENTISSAGE, entrees)
    return {nom: rul_ml.entrainer(nom, X, y, groupes)[0] for nom in ["SVR", "GPR", "Random Forest"]}, X, y, groupes


def covariables_ml(lisse, selection, beta, eta, parametres_wiener):
    print("3b. Covariables : modèles ML avec la charge et la vitesse en entrée")
    entrees_base = selection + ["depuis_fpt_min"]
    entrees_cov = entrees_base + ["charge_kN", "vitesse_rpm"]
    bilans, modeles_base = {}, None
    for nom_version, entrees in [("sans covariables", entrees_base), ("avec charge et vitesse", entrees_cov)]:
        modeles, X, y, groupes = entrainer_ml(lisse, entrees)
        if modeles_base is None:
            modeles_base, jeu_base = modeles, (X, y, groupes)
        evaluation = pipeline.etape_evaluation(lisse, entrees, selection, beta, eta, parametres_wiener, modeles, None)
        bilans[nom_version] = {m["modele"]: {k: m[k] for k in ("rmse_min", "mae_min", "score_phase", "score_challenge")}
                               for m in evaluation["metriques"] if m["modele"] in modeles}
    for nom in ["SVR", "GPR", "Random Forest"]:
        a, b = bilans["sans covariables"][nom], bilans["avec charge et vitesse"][nom]
        print(f"   {nom:14s} RMSE {a['rmse_min']:.2f} -> {b['rmse_min']:.2f} min")
    return bilans, modeles_base, entrees_base, jeu_base


# ---------------------------------------------------------------- 4. données simulées
def simulation_weibull(beta, eta, ages_censure, n_repetitions=400, facteurs=(1, 3, 12), graine=1):
    """On tire des durées de vie dans une Weibull CONNUE, avec le même plan d'essai que PRONOSTIA
    (6 essais complets + 11 censurés aux mêmes âges), puis on réestime beta et le B10."""
    print("4a. Données simulées : la Weibull retrouve-t-elle beta et le B10 ?")
    rng = np.random.default_rng(graine)
    resultats = {}
    for facteur in facteurs:                       # facteur 3 = le même plan d'essai répété 3 fois (51 roulements)
        n_complets, censures = 6 * facteur, np.tile(ages_censure, facteur)
        betas, b10s = [], []
        for _ in range(n_repetitions):
            vies = eta * rng.weibull(beta, n_complets + len(censures))
            limites = np.concatenate([np.full(n_complets, np.inf), censures])
            observees = np.minimum(vies, limites)
            evenements = (vies <= limites).astype(int)
            if evenements.sum() < 2:
                continue
            b, e = reliability.ajuster_weibull(observees, evenements)
            betas.append(b)
            b10s.append(reliability.quantile_vie(0.10, b, e))
        n = n_complets + len(censures)
        resultats[str(n)] = {"n_roulements": n, "beta": resume(betas), "b10_min": resume(b10s, 1),
                             "betas": arrondir(betas, 3)}
        print(f"   {n:3d} roulements : beta estimé entre {resultats[str(n)]['beta']['p05']} et "
              f"{resultats[str(n)]['beta']['p95']} (vrai : {beta:.2f})")
    return {"beta_vrai": round(beta, 3), "b10_vrai_min": round(float(reliability.quantile_vie(0.1, beta, eta)), 1),
            "par_taille": resultats}


def trajectoire(forme, parametres, rng, seuils_reels=None, pas=10 / 60):
    """Une trajectoire de HI simulée depuis le FPT jusqu'au seuil (vraie RUL connue).
    seuils_reels = (moyenne, écart-type) du vrai seuil, s'il diffère de celui que le modèle croit."""
    moyenne, ecart = seuils_reels or (parametres["seuil"], parametres["seuil_ecart"])
    seuil = max(rng.normal(moyenne, ecart), 0.5)
    if forme != "exponentielle (palier puis accélération)":
        mu = -1.0
        while mu < 0.05:                                          # dérive tirée dans la loi a priori
            mu = rng.normal(parametres["mu0"], parametres["s0"])
        hi = [0.0]
        while hi[-1] < seuil and len(hi) < 6 * 600:
            hi.append(hi[-1] + mu * pas + parametres["sigma"] * np.sqrt(pas) * rng.normal())
        hi = np.array(hi)
    else:                                                         # palier puis accélération (comme les vrais roulements)
        duree = rng.uniform(10, 120)
        t = np.arange(0, duree, pas)
        hi = seuil * (np.exp(4 * t / duree) - 1) / (np.exp(4) - 1) + rng.normal(0, 0.15, len(t))
    temps = np.arange(len(hi)) * pas
    return temps, hi, temps[-1] - temps


def simulation_wiener(parametres, seuils_test, n_roulements=120, graine=2):
    """3 scénarios : (1) les données suivent exactement le modèle ; (2) le HI a un palier puis accélère ;
    (3) le vrai seuil de défaillance varie autant que sur les roulements de test (bien plus qu'à l'apprentissage)."""
    print("4b. Données simulées : l'intervalle à 90 % du Wiener contient-il 90 % des vraies RUL ?")
    rng = np.random.default_rng(graine)
    resultats = {}
    scenarios = {"linéaire (comme le modèle)": None, "exponentielle (palier puis accélération)": None,
                 "seuil dispersé (comme les roulements de test)": seuils_test}
    for forme, seuils_reels in scenarios.items():
        couvertures, rmses = [], []
        for _ in range(n_roulements):
            temps, hi, vrai = trajectoire(forme, parametres, rng, seuils_reels)
            garder = slice(0, None, 6)                            # une prédiction par minute (plus rapide)
            mediane, bas, haut = rul_wiener.predire(temps[garder], hi[garder], parametres, n_tirages=1000)
            v = vrai[garder]
            couvertures.append(np.mean((v >= bas) & (v <= haut)))
            rmses.append(ev.rmse(v, mediane))
        resultats[forme] = {"couverture_90": round(float(np.mean(couvertures)), 3),
                            "rmse_min": round(float(np.mean(rmses)), 2)}
        print(f"   {forme:46s} couverture = {resultats[forme]['couverture_90']:.0%}")
    resultats["seuil_test_observe"] = {"moyenne": round(seuils_test[0], 2), "ecart_type": round(seuils_test[1], 2),
                                       "ecart_type_apprentissage": round(parametres["seuil_ecart"], 2)}
    return resultats


# ---------------------------------------------------------------- 5. sensibilité aux paramètres
def sensibilite_fpt(features, roulements):
    print("5a. Sensibilité : seuil du FPT de 2 à 5 sigmas")
    lisse = health.lisser(health.normaliser(features))
    vies = dict(zip(roulements["roulement"], roulements["duree_vie_s"] / 60))
    lignes = []
    for n_sigma in [2, 3, 4, 5]:
        health.N_SIGMA = n_sigma
        avec_fpt = health.ajouter_fpt(lisse)
        fausses, delais, degrades, non_detectes = 0, [], 0, 0
        for roulement, groupe in avec_fpt.groupby("roulement"):
            fpt = int(groupe["fpt"].iloc[0])
            episodes = pipeline.episodes_alarme(groupe["fpt_courant"].to_numpy())
            fausses += len(episodes) - (1 if fpt > 0 else 0)
            if fpt == 0:
                non_detectes += 1
                continue
            delais.append(vies[roulement] - fpt * config.PAS_TEMPS / 60)
            if roulement in config.TEST:
                k = config.TEST[roulement]["troncature"]
                etat = int(groupe["fpt_courant"].iloc[k - 1])
                degrades += int(etat > 0 and etat == fpt)
        lignes.append({"n_sigma": n_sigma, "fausses_alarmes": int(fausses), "non_detectes": int(non_detectes),
                       "delai_alerte_median_min": round(float(np.median(delais)), 1),
                       "delai_alerte_moyen_min": round(float(np.mean(delais)), 1),
                       "test_degrades_a_la_troncature": int(degrades)})
        print(f"   {n_sigma} sigmas : {fausses:2d} fausses alarmes, délai d'alerte médian "
              f"{lignes[-1]['delai_alerte_median_min']} min")
    health.N_SIGMA = 3
    return lignes


def sensibilite_wiener(lisse, parametres):
    print("5b. Sensibilité : variantes du modèle de Wiener")
    variantes = {"modèle retenu": parametres,
                 "seuil fixe (sans incertitude)": {**parametres, "seuil_ecart": 0.0},
                 "a priori très large (s0 x 10)": {**parametres, "s0": parametres["s0"] * 10}}
    lignes = []
    for nom, p in variantes.items():
        rmses, couvertures = [], []
        for roulement in config.TEST:
            groupe = lisse[lisse["roulement"] == roulement]
            post = groupe[groupe["index"] >= groupe["fpt"].iloc[0]]
            mediane, bas, haut = rul_wiener.predire(post["temps_s"].to_numpy() / 60, post["hi"].to_numpy(), p)
            vrai = post["rul_s"].to_numpy() / 60
            rmses.append(ev.rmse(vrai, mediane))
            couvertures.append(np.mean((vrai >= bas) & (vrai <= haut)))
        lignes.append({"variante": nom, "rmse_min": round(float(np.mean(rmses)), 2),
                       "couverture_90": round(float(np.mean(couvertures)), 3)})
        print(f"   {nom:32s} RMSE {lignes[-1]['rmse_min']:.2f} min, couverture {lignes[-1]['couverture_90']:.0%}")
    return lignes


def grille_svr(X, y, groupes):
    print("5c. Sensibilité : grille d'hyperparamètres du SVR")
    modele, grille = rul_ml.modeles_et_grilles()["SVR"]
    recherche = GridSearchCV(modele, grille, cv=LeaveOneGroupOut(), scoring="neg_mean_absolute_error", n_jobs=-1)
    recherche.fit(X, y, groups=groupes)
    lignes = []
    for params, score in zip(recherche.cv_results_["params"], recherche.cv_results_["mean_test_score"]):
        lignes.append({"C": params["svr__C"], "gamma": str(params["svr__gamma"]), "epsilon": params["svr__epsilon"],
                       "mae_validation_log": round(float(-score), 4)})
    return lignes


# ---------------------------------------------------------------- 6. explicabilité (XAI)
NOMS_LISIBLES = {"depuis_fpt_min": "temps depuis le FPT", "charge_kN": "charge", "vitesse_rpm": "vitesse"}


def nom_lisible(colonne):
    """'v_wpt_5' -> 'vertical, bande 6,4–8,0 kHz'."""
    if colonne in NOMS_LISIBLES:
        return NOMS_LISIBLES[colonne]
    axe, nom = ("horizontal" if colonne[0] == "h" else "vertical"), colonne[2:]
    if nom.startswith("wpt_"):
        k = int(nom[4:])
        return f"{axe}, bande {1.6 * (k - 1):.1f}–{1.6 * k:.1f} kHz".replace(".", ",")
    return f"{axe}, {nom}"


def explicabilite(lisse, modeles, entrees, n_repetitions=10, graine=0):
    print("6. Explicabilité : importance des entrées par permutation")
    rng = np.random.default_rng(graine)
    test = lisse[lisse["roulement"].isin(list(config.TEST)) & (lisse["depuis_fpt_min"] >= 0)]
    X, vrai = test[entrees].to_numpy(), test["rul_s"].to_numpy() / 60
    resultats = {}
    for nom, modele in modeles.items():
        erreur_base = ev.mae(vrai, rul_ml.predire(modele, X))
        importances = []
        for j, colonne in enumerate(entrees):
            hausses = []
            for _ in range(n_repetitions):
                melange = X.copy()
                rng.shuffle(melange[:, j])                       # on casse le lien entre cette entrée et la RUL
                hausses.append(ev.mae(vrai, rul_ml.predire(modele, melange)) - erreur_base)
            importances.append({"entree": colonne, "nom": nom_lisible(colonne),
                                "hausse_mae_min": round(float(np.mean(hausses)), 3),
                                "ecart_type": round(float(np.std(hausses)), 3)})
        resultats[nom] = {"mae_base_min": round(erreur_base, 2),
                          "importances": sorted(importances, key=lambda d: -d["hausse_mae_min"])}
        print(f"   {nom:14s} entrée la plus utile : {resultats[nom]['importances'][0]['nom']}")
    foret = modeles["Random Forest"]
    resultats["impurete_random_forest"] = sorted(
        [{"entree": c, "nom": nom_lisible(c), "importance": round(float(v), 4)}
         for c, v in zip(entrees, foret.feature_importances_)], key=lambda d: -d["importance"])
    resultats["n_points_test"] = int(len(test))
    return resultats


# ---------------------------------------------------------------- 7. tests statistiques
def tests_statistiques(rul):
    print("7. Tests statistiques sur les RMSE par roulement")
    par = rul["par_roulement"]
    reference = np.array(par["Wiener"]["rmse_min"])
    lignes = []
    for nom in rul["modeles"]:
        if nom == "Wiener":
            continue
        autre = np.array(par[nom]["rmse_min"])
        p = float(stats.wilcoxon(reference, autre).pvalue)
        lignes.append({"modele": nom, "wiener_meilleur_sur": int(np.sum(reference < autre)),
                       "difference_mediane_min": round(float(np.median(autre - reference)), 2),
                       "p_valeur": round(p, 4), "significatif_5pct": p < 0.05})
    friedman = stats.friedmanchisquare(*[par[nom]["rmse_min"] for nom in rul["modeles"]])
    print(f"   Friedman (7 modèles) : p = {friedman.pvalue:.4f}")
    return {"wilcoxon_contre_wiener": lignes,
            "friedman": {"statistique": round(float(friedman.statistic), 3), "p_valeur": round(float(friedman.pvalue), 4)}}


# ---------------------------------------------------------------- 8. économie
def analyse_economique(roulements, sante, rul, beta, eta):
    print("8. Économie : stratégies de maintenance sur les 11 roulements de test")
    p = economie.PARAMETRES
    vies = [float(roulements.set_index("roulement").loc[r, "duree_vie_s"]) / 60 for r in config.TEST]
    fpt = [sante["roulements"][r]["fpt_index"] * config.PAS_TEMPS / 60 for r in config.TEST]
    fausses = [len(sante["roulements"][r]["episodes"]) - 1 for r in config.TEST]
    courbes = [rul["courbes"][r] for r in config.TEST]
    modeles = [m for m in rul["modeles"] if m != "Fiabilité (B10)"]
    strategies, t_etoile = economie.comparer_strategies(vies, fpt, courbes, fausses, beta, eta, **p, modeles=modeles)
    for nom, bilan in strategies.items():
        print(f"   {nom:34s} {bilan['cout_par_heure_u']:.3f} u/h  pannes {bilan['pannes']:2d}/11  "
              f"gain {bilan['gain_vs_corrective_pct']:+.0f} %")

    ages, couts = economie.couts_age(beta, eta, p["cp"], p["cc"], t_max=700)
    mttf = reliability.mttf(beta, eta)
    rapports = []
    for rapport in [1.5, 2, 3, 5, 7, 10]:                       # sensibilité au rapport Cc / Cp
        t_opt, c_opt = economie.age_optimal(beta, eta, 1.0, rapport)
        c_corrective = rapport / mttf * 60
        rapports.append({"cc_sur_cp": rapport, "t_etoile_min": round(t_opt, 1),
                         "gain_theorique_pct": round(100 * (1 - c_opt / c_corrective), 1)})
    seuils = []
    for seuil in [2, 5, 10, 15, 20, 30, 45, 60]:                # sensibilité au seuil de décision s
        bilan, _ = economie.comparer_strategies(vies, fpt, courbes, fausses, beta, eta, cp=p["cp"], cc=p["cc"],
                                                ci=p["ci"], delai_min=p["delai_min"], seuil_min=seuil, modeles=modeles)
        seuils.append({"seuil_min": seuil, **{nom.replace("Prédictive (", "").rstrip(")"): b["cout_par_heure_u"]
                                              for nom, b in bilan.items() if nom.startswith("Prédictive")}})
    return {"parametres": p, "t_etoile_min": round(t_etoile, 1), "strategies": strategies,
            "courbe_age": {"age_min": arrondir(ages[::20], 1), "cout_par_heure_u": arrondir(couts[::20], 4)},
            "sensibilite_rapport_couts": rapports, "sensibilite_seuil": seuils,
            "vies_min": arrondir(vies, 1), "fausses_alarmes": fausses, "roulements": list(config.TEST)}


# ---------------------------------------------------------------- KPIs
def indicateurs_cles(fiabilite_ref, bootstrap, sante, rul, roulements):
    w = fiabilite_ref["weibull"]
    vies = dict(zip(roulements["roulement"], roulements["duree_vie_s"] / 60))
    delais = [vies[r] - info["fpt_min"] for r, info in sante["roulements"].items()]
    fausses = [len(info["episodes"]) - 1 for info in sante["roulements"].values()]
    meilleur = min(rul["metriques"], key=lambda m: m["rmse_min"])
    return {
        "mttf_min": w["mttf_min"], "b10_min": w["b10_min"], "b50_min": w["b50_min"],
        "b10_ic95": bootstrap["ic95"]["b10_min"], "b50_ic95": bootstrap["ic95"]["b50_min"],
        "taux_defaillance_300min_par_h": round(float(60 * reliability.taux_defaillance(300, w["beta"], w["eta_min"])), 4),
        "delai_alerte_median_min": round(float(np.median(delais)), 1),
        "roulements_alerte_plus_5min": int(np.sum(np.array(delais) > 5)),
        "fausses_alarmes_total": int(sum(fausses)),
        "roulements_avec_fausse_alarme": int(np.sum(np.array(fausses) > 0)),
        "meilleur_rmse": {"modele": meilleur["modele"], "rmse_min": meilleur["rmse_min"]},
        "couverture_wiener_90": rul["couverture_wiener_90"]["moyenne"],
    }


def main():
    debut = time.time()
    features = pd.read_csv(config.FICHIER_FEATURES)
    roulements = pd.read_csv(config.DOSSIER_TRAITE / "roulements.csv")
    rul, sante, fiabilite_ref = lire_resultat("rul.json"), lire_resultat("sante.json"), lire_resultat("fiabilite.json")
    beta, eta = fiabilite_ref["weibull"]["beta"], fiabilite_ref["weibull"]["eta_min"]
    durees, evenements = reliability.donnees_survie(roulements)

    complements = {"vraisemblance": vraisemblance(durees, evenements)}
    complements["bootstrap_fiabilite"] = intervalles_bootstrap(durees, evenements)
    complements["covariables_fiabilite"] = covariables_fiabilite(roulements, durees, evenements)

    lisse, selection, _ = pipeline.etape_sante(features)
    lisse = rul_ml.ajouter_depuis_fpt(lisse)
    lisse["charge_kN"] = lisse["roulement"].map(lambda r: config.CONDITIONS[config.condition_de(r)]["charge_N"] / 1000)
    lisse["vitesse_rpm"] = lisse["roulement"].map(lambda r: config.CONDITIONS[config.condition_de(r)]["vitesse_rpm"])
    parametres_wiener = rul_wiener.estimer_parametres(lisse, config.APPRENTISSAGE)
    bilans, modeles_base, entrees_base, (X, y, groupes) = covariables_ml(lisse, selection, beta, eta, parametres_wiener)
    complements["covariables_ml"] = bilans

    complements["simulation_weibull"] = simulation_weibull(beta, eta, durees[evenements == 0])
    finaux_test = [lisse[lisse["roulement"] == r]["hi"].tail(5).mean() for r in config.TEST]   # HI à la panne
    complements["simulation_wiener"] = simulation_wiener(parametres_wiener,
                                                         (float(np.mean(finaux_test)), float(np.std(finaux_test))))
    complements["sensibilite_fpt"] = sensibilite_fpt(features, roulements)
    complements["sensibilite_wiener"] = sensibilite_wiener(lisse, parametres_wiener)
    complements["grille_svr"] = grille_svr(X, y, groupes)
    complements["explicabilite"] = explicabilite(lisse, modeles_base, entrees_base)
    complements["tests_statistiques"] = tests_statistiques(rul)
    complements["economie"] = analyse_economique(roulements, sante, rul, beta, eta)
    complements["kpis"] = indicateurs_cles(fiabilite_ref, complements["bootstrap_fiabilite"], sante, rul, roulements)
    complements["duree_s"] = round(time.time() - debut, 1)

    pipeline.ecrire_json("complements.json", complements)
    print(f"\nTerminé en {time.time() - debut:.0f} s")


if __name__ == "__main__":
    main()

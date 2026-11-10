# Étude comparative de pronostic de défaillance de roulements

**Module :** Sûreté de fonctionnement et maintenance · **Encadrant :** M. Fayçal Ben Hmida · ENSIT, GMAM 3
**Équipe :** Farah Belghith, Rami Kammoun, Selim Najjar
**Données :** FEMTO-ST / PRONOSTIA, challenge IEEE PHM 2012 (17 roulements, 3 conditions opératoires)
**Livrables :** pipeline Python en 4 sprints, 23 figures, dashboard interactif en 7 pages (Streamlit)

**Question étudiée :** à partir des vibrations d'un roulement, peut-on prédire sa durée de vie résiduelle (RUL, *Remaining Useful Life*) ? Et quelle famille de modèles marche le mieux : fiabilité de la population, modèle stochastique, machine learning classique ou deep learning ?

---

## 1. Lancer le dashboard (2 minutes)

Les résultats sont **déjà calculés** (dossier `results/`). Il suffit de Python et de 4 bibliothèques.
Dans VS Code : *Fichier → Ouvrir le dossier* (`Projet_FM`), puis *Terminal → Nouveau terminal* :

```bash
pip install streamlit plotly pandas numpy
streamlit run dashboard/app.py
```

Le dashboard s'ouvre sur **http://localhost:8501**. Lancer la commande depuis le dossier du projet (pour que le thème de `.streamlit/config.toml` soit appliqué).

### Recalculer les résultats (optionnel)

```bash
pip install -r requirements.txt        # scipy, scikit-learn, PyWavelets, lifelines, matplotlib, tensorflow…
python 2_lancer_pipeline.py            # sprints 2 à 4 -> results/   (7 à 10 min, surtout le LSTM et le CNN 1D)
python 2_lancer_pipeline.py --rapide   # 1 à 3 min : sans recherche d'hyperparamètres pour le deep learning
python 4_analyses_complementaires.py   # vraisemblance, covariables, données simulées, XAI, économie (~5 min)
python 3_figures_rapport.py            # -> figures/ (23 figures PNG, 300 dpi, ~15 s)
```

- TensorFlow est **optionnel** : s'il n'est pas installé, l'étape 2 tourne quand même, sans le LSTM ni le CNN 1D.
- Chaque exécution écrit `results/run_info.json` : date, durée, versions des bibliothèques et graine aléatoire (42).
- Tout est déterministe (graines fixées). Seuls les réseaux de neurones peuvent varier légèrement d'une machine à l'autre (calcul flottant de TensorFlow).
- Si le dashboard est ouvert pendant un nouveau calcul, il relit tout seul les nouveaux résultats.

### Repartir des fichiers bruts (optionnel, ~2,5 Go)

Le fichier `data/processed/features.csv` (les features des 24 889 fichiers) est fourni, donc cette étape n'est utile que pour tout refaire depuis le début. Les données brutes viennent d'un miroir GitHub du jeu de données officiel. On ne télécharge que les fichiers d'accélération :

```bash
git clone --filter=blob:none --sparse https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset data/raw
cd data/raw
git sparse-checkout set --no-cone "/Learning_set/*/acc_*" "/Full_Test_Set/*/acc_*"
cd ../..
python 1_extraire_features.py          # -> data/processed/   (3 à 10 min selon le PC)
```

### Lancer les tests

```bash
python -m pytest                       # 29 tests, environ 1 minute
```

Les tests vérifient les fonctions sur des cas dont on connaît la réponse à la main : score du challenge, RMS d'une sinusoïde, kurtosis d'un bruit gaussien (= 3), bande d'ondelettes d'une sinusoïde à 2,4 kHz, détection du FPT et annulation d'un transitoire, Weibull avec β = 1 (loi exponentielle), B10, L10, processus de Wiener à dérive constante, coût d'une stratégie de maintenance. Ils ouvrent aussi chacune des 7 pages du dashboard.

---

## 2. Structure du projet (plan de sprints)

```
data/raw/                    données brutes FEMTO (à télécharger, voir plus haut ; non versionnées)
data/processed/              features.csv (36 features x 24 889 fichiers), roulements.csv, signaux.json
src/
  config.py                  constantes : chemins, 25,6 kHz, conditions, troncatures et RUL officielles
  data.py          Sprint 1  lecture des fichiers acc_*.csv, tableau des roulements, contrôle qualité
  preprocessing.py Sprint 1  filtre passe-bande Butterworth 10 Hz – 10 kHz (sans déphasage)
  reliability.py   Sprint 1  Kaplan-Meier, Weibull avec censure, MTTF, B10, L10, durée de vie résiduelle
  features.py      Sprint 2  18 features par axe : temporelles, fréquentielles, paquets d'ondelettes
  health.py        Sprint 2  normalisation, FPT (règle des 3 σ), sélection des features, indicateur de santé
  rul_wiener.py    Sprint 3  processus de Wiener bayésien (dérive et seuil incertains)
  rul_ml.py        Sprint 3  SVR, GPR, Random Forest et recherche d'hyperparamètres
  rul_deep.py      Sprint 3  LSTM et CNN 1D (Keras)
  evaluation.py    Sprint 4  RMSE, MAE, score PHM asymétrique, intervalles bootstrap
  economie.py      Sprint 4  coût par heure des stratégies de maintenance, âge optimal T*
1_extraire_features.py       étape 1 : fichiers bruts -> data/processed/
2_lancer_pipeline.py         étape 2 : sprints 2, 3 et 4 -> results/
3_figures_rapport.py         étape 3 : results/ -> figures/
4_analyses_complementaires.py étape 4 : vraisemblance, covariables, données simulées, sensibilité, XAI, tests, économie
results/                     JSON et CSV lus par le dashboard et par le script des figures (complements.json : étape 4)
figures/                     les 23 figures du rapport
dashboard/                   app.py (navigation), commun.py (chargement, couleurs, style), pages/ (7 pages)
tests/                       test_projet.py (fonctions), test_dashboard.py (chaque page du dashboard)
rapport/                     rapport LaTeX (main.tex, chapitres/, references.bib pour Zotero) et le PDF compilé
```

---

## 3. Méthodologie et résultats

### 3.1 Données et prétraitement (Sprint 1)

La plateforme PRONOSTIA use des roulements jusqu'à la défaillance, de façon accélérée. Deux accéléromètres (horizontal et vertical) enregistrent 0,1 s de signal (2 560 points à 25,6 kHz) toutes les 10 s. L'essai s'arrête quand la vibration dépasse 20 g.

| Condition | Vitesse | Charge | Apprentissage (essais complets) | Test (tronqués) |
|---|---|---|---|---|
| 1 | 1 800 tr/min | 4 000 N | Bearing1_1, Bearing1_2 | Bearing1_3 à Bearing1_7 |
| 2 | 1 650 tr/min | 4 200 N | Bearing2_1, Bearing2_2 | Bearing2_3 à Bearing2_7 |
| 3 | 1 500 tr/min | 5 000 N | Bearing3_1, Bearing3_2 | Bearing3_3 |

- **Contrôle qualité :** aucun fichier incomplet ni valeur manquante. Bearing1_1 a 3 horodatages incohérents : le temps est reconstruit à partir du numéro de fichier (10 s par fichier). Pour Bearing1_4, la RUL officielle (339 s) ne correspond pas au nombre de fichiers (2 890 s) : c'est une anomalie connue du jeu de données, on garde la valeur officielle pour le score du challenge.
- **Filtrage :** suppression de la tendance, puis filtre passe-bande Butterworth d'ordre 4 entre 10 Hz et 10 kHz, appliqué dans les deux sens (aucun déphasage).
- **Découpage par roulement :** les 6 roulements d'apprentissage servent à entraîner les modèles et à choisir les hyperparamètres (validation « un roulement de côté »). Les 11 roulements de test ne sont jamais vus pendant l'apprentissage. Aucun roulement n'est dans les deux ensembles.
- **Normalisation multi-conditions :** chaque feature est divisée par son niveau à l'état sain (fichiers 31 à 150, soit de 5 à 25 min, après le rodage), puis on prend le logarithme. 0 veut dire « comme à l'état sain », quelle que soit la charge ou la vitesse.

### 3.2 Fiabilité de la population (Sprint 1)

On traite les 6 roulements d'apprentissage comme des **défaillances** et les 11 roulements de test comme des **données censurées à droite** (ils fonctionnaient encore à la troncature).

| Indicateur | Résultat |
|---|---|
| Weibull 2P par maximum de vraisemblance avec censure | **β = 1,85**, η = 422 min |
| MTTF = η·Γ(1 + 1/β) | 375 min |
| B10 / B50 | 125 min / 346 min |
| L10 du catalogue, (C/P)³ millions de tours avec C = 4 000 N | 556 / 524 / 341 min (conditions 1 / 2 / 3) |

- β > 1 : le taux de défaillance augmente avec l'âge, c'est de l'**usure**.
- La plupart des roulements défaillent bien avant leur L10 : la formule des catalogues ne s'applique pas à ces essais accélérés, comme l'indiquent les organisateurs du challenge.
- **B10 conditionnel :** pour un roulement sain d'âge *a*, c'est la durée *x* qu'il tiendra encore avec 90 % de chances, soit R(a + x) / R(a) = 0,9. C'est la prédiction utilisée tant qu'aucun défaut n'est détecté. Avec la durée de vie résiduelle moyenne (MRL) à la place, le score du challenge tomberait à 0 : la moyenne prédit presque toujours trop tard.

### 3.3 Premier défaut, features et indicateur de santé (Sprint 2)

**Features (18 par axe) :**
- temporelles : RMS, pic, crête à crête, kurtosis, asymétrie, facteurs de crête, de forme et d'impulsion
- fréquentielles : fréquence moyenne et entropie spectrale
- temps-fréquence : énergie des 8 sous-bandes (1,6 kHz chacune) d'une décomposition en paquets d'ondelettes (db4, niveau 3)

**Détection du premier défaut (FPT, *First Predicting Time*) par la règle des 3 σ :**
- seuil de chaque axe = moyenne + 3 écarts-types du RMS lissé (1 min) à l'état sain
- alarme après 3 mesures consécutives au-dessus du seuil
- alarme annulée si le RMS repasse 5 min sous le seuil (transitoire)
- la règle n'utilise que le passé : elle fonctionnerait en temps réel

À la troncature du challenge, **8 roulements de test sur 11 ne montrent encore aucun défaut**. Seuls Bearing1_3, Bearing1_4 et Bearing3_3 sont déjà dégradés.

**Sélection des features :** trois critères entre 0 et 1, calculés sur la phase de dégradation des 6 roulements d'apprentissage :
- monotonie (la feature évolue toujours dans le même sens)
- corrélation de Spearman avec le temps (tendance régulière)
- pronosticabilité de Coble et Hines (tous les roulements finissent à peu près à la même valeur)

Les 8 meilleures, surtout des énergies d'ondelettes (`v_wpt_5`, `h_wpt_3`, `v_wpt_4`, `v_wpt_6`, `h_wpt_7`, `h_wpt_6`, `h_wpt_4`, `v_wpt_2`), sont les **entrées des modèles ML et deep learning**.

**Indicateur de santé (HI) :** on essaie la moyenne des *k* meilleures features, pour *k* = 1 à 10, et on garde le *k* au meilleur score. Résultat : **k = 1**, le HI est `v_wpt_5` (énergie de la bande 6,4 – 8 kHz sur l'axe vertical). Son score est de 0,644, contre 0,599 pour la moyenne des 8 meilleures : moyenner rend la fin de vie plus régulière mais casse la tendance. Ici, la sélection l'emporte sur la fusion.

- Seuil de défaillance D = 3,34 ± 0,63 : moyenne et écart-type du HI en fin de vie des roulements d'apprentissage.
- RUL cible : linéaire décroissante du FPT jusqu'à la défaillance (en minutes). Avant le FPT, le roulement est jugé sain et sa RUL vient de la fiabilité.

### 3.4 Modèles de RUL (Sprint 3)

| Famille | Modèle | Réglages retenus | Combinaisons testées | Temps d'apprentissage |
|---|---|---|---|---|
| Fiabilité (sans capteur) | B10 conditionnel | Weibull β = 1,85, η = 422 min | — | — |
| Stochastique | Processus de Wiener bayésien | dérive a priori 0,281 ± 0,276 /min, σ = 0,245 | — | < 0,1 s |
| ML classique | SVR (noyau RBF) | C = 1, γ = 0,1, ε = 0,05 | 24 | 6,5 s |
| ML classique | GPR (RBF + bruit blanc) | longueur de corrélation 1,0 | 3 | 6,6 s |
| ML classique | Random Forest (300 arbres) | profondeur libre, 50 % des features, 5 exemples par feuille | 18 | 14 s |
| Deep learning | LSTM (5 921 paramètres) | 32 unités, fenêtre de 6 mesures | 4 | 200 s |
| Deep learning | CNN 1D (1 521 paramètres) | 16 filtres, fenêtre de 12 mesures | 4 | 142 s |

Temps mesurés sur le PC de l'équipe (Windows 10, Python 3.10), qui a produit les résultats de référence de `results/`.

- **Wiener :** HI(t) = HI(FPT) + μ·t + σ·B(t). La loi a priori de la dérive μ est apprise sur les roulements d'apprentissage, puis mise à jour (loi normale conjuguée) avec les mesures du roulement suivi. On tire 2 000 couples (dérive, seuil). Le temps de premier passage du seuil suit une loi inverse gaussienne : sa médiane donne la RUL, ses quantiles 5 % et 95 % un intervalle de prédiction à 90 %.
- **ML et deep learning :** entrées = les 8 features + le temps écoulé depuis le FPT ; cible = log(1 + RUL en minutes), pour que les erreurs proches de la fin de vie comptent davantage ; 100 exemples par roulement d'apprentissage, pour que chaque roulement pèse autant. Le LSTM et le CNN 1D voient une fenêtre glissante des dernières mesures.
- **Hyperparamètres :** recherche sur grille, avec une validation croisée qui laisse de côté un roulement d'apprentissage à la fois (`GridSearchCV` + `LeaveOneGroupOut`). Les roulements de test ne servent jamais à régler les modèles.

### 3.5 Évaluation comparative (Sprint 4)

Deux protocoles sur les 11 roulements de test :
1. **Phase de dégradation** (du FPT à la défaillance) : RMSE et MAE en minutes par roulement, puis moyenne ; score PHM calculé à 25 %, 50 % et 75 % de la phase de chaque roulement.
2. **Protocole officiel du challenge** : une seule prédiction à l'instant de troncature. Erreur %Er = 100 × (RUL réelle − RUL prédite) / RUL réelle. Le score pénalise beaucoup plus une prédiction tardive (score 0,5 à 5 % de retard) qu'une prédiction précoce (0,5 à 20 % d'avance).

| Modèle | RMSE (min) | MAE (min) | Score PHM (phase) | Score challenge |
|---|---|---|---|---|
| Fiabilité (B10) | 44,4 | 42,8 | 0,044 | 0,170 |
| **Wiener** | **16,4** | **13,0** | 0,059 | **0,194** |
| SVR | 20,0 | 17,1 | 0,059 | 0,175 |
| GPR | 18,3 | 15,3 | 0,064 | 0,171 |
| Random Forest | 23,0 | 18,1 | 0,059 | 0,169 |
| **LSTM** | 18,7 | 15,2 | **0,084** | 0,187 |
| CNN 1D | 17,9 | 15,3 | 0,063 | 0,169 |

**Lecture des résultats.**
- Après le FPT, **tous les modèles fondés sur les capteurs battent la fiabilité seule** (RMSE de 16 à 23 min, contre 44 min).
- Le **processus de Wiener** est le plus précis (RMSE, MAE) et le meilleur selon le protocole officiel. Il est simple, explicable et donne un intervalle de prédiction.
- Le **LSTM** a le meilleur score PHM : il prédit moins souvent trop tard. Il coûte en revanche 30 fois plus de calcul que le SVR.
- **Incertitude :** les intervalles de confiance bootstrap (95 %) se chevauchent largement, par exemple RMSE du Wiener [6,3 ; 28,0] et score PHM du LSTM [0,023 ; 0,155]. Le test de Friedman (p = 0,022) ne distingue que la fiabilité seule ; entre les modèles fondés sur les capteurs, le test de Wilcoxon ne trouve aucun écart significatif (p de 0,21 à 0,70).
- **Protocole du challenge :** 8 roulements sur 11 sont encore sains à la troncature, donc tous les modèles y donnent la même prédiction (B10 conditionnel). Les écarts ne viennent que de 3 roulements.
- Pour situer : la meilleure méthode recensée par Dhungana et al. (2025) atteint un score de 0,791 sur les 11 roulements, avec des features et des modèles beaucoup plus travaillés.

**Limites.**
- Très peu de données : 6 roulements d'apprentissage, et des durées de vie très variables (de 38 à 467 min), même sous une même condition.
- Pour la plupart des roulements de test, la défaillance est brutale : la phase de dégradation détectable dure de 1 à 17 min, sauf pour Bearing1_3 (177 min) et Bearing1_4 (56 min).
- L'intervalle à 90 % du Wiener ne contient la RUL réelle que dans **54 %** des cas : l'incertitude est sous-estimée. Les données simulées montrent pourquoi : le HI des roulements de test finit à des niveaux bien plus dispersés (écart-type 1,57) que sur l'apprentissage (0,63), et le seuil de défaillance D appris sur 6 roulements est trop sûr de lui.

### 3.6 Analyses complémentaires (étape 4)

| Analyse | Résultat principal |
|---|---|
| Vraisemblance | 4 lois comparées (AIC) : Weibull, log-normale et log-logistique sont équivalentes (écart < 2) ; test β = 1 : p = 0,097 (6 défaillances seulement) |
| B10 et B50 | 125 min [78 ; 234] et 346 min [219 ; 573] (IC 95 % par bootstrap) |
| Covariables | +1 kN de charge -> durée de vie × 0,40 (p = 0,042) ; B10 de 175 min (condition 1) à 69 min (condition 3) |
| Données simulées | β retrouvé en moyenne, mais entre 1,36 et 3,34 avec 17 roulements ; intervalle du Wiener juste (92 %) quand ses hypothèses sont vraies, 63 % quand le seuil est aussi dispersé que sur le test |
| Sensibilité | seuil du FPT de 2 à 5 σ : de 25 à 11 fausses alarmes, délai d'alerte médian de 8,0 à 6,8 min |
| Explicabilité (XAI) | les modèles ML s'appuient surtout sur les énergies de bandes d'ondelettes (1,6 à 9,6 kHz), pas seulement sur le temps depuis le FPT |
| Économie | coût par heure : correctif 1,14 u/h, préventif à T* = 224 min 0,84, conditionnel 0,64, **prédictif Wiener 0,60 (−47 %)** |

**Perspectives.** Modèle de dégradation exponentiel ou filtre particulaire, seuil de défaillance appris par condition, intervalles calibrés (prédiction conforme), GRU et TCN, apprentissage par transfert entre conditions opératoires (voir Dhungana et al., 2025), autres covariables (température).

---

## 4. Pages du dashboard

Le dashboard est en thème sombre (noir et bleu) : chaque page a sa nuance de bleu et commence par un bandeau qui résume ce qu'elle montre.

1. **Vue d'ensemble :** chiffres clés, la démarche en cartes de couleur, durées de vie des 17 roulements, classement des modèles avec intervalles de confiance.
2. **Données et fiabilité :**
   - tableau des 17 roulements et contrôle qualité
   - signal brut et filtré d'un roulement au choix, évolution du RMS et du kurtosis
   - Kaplan-Meier et Weibull, taux de défaillance λ(t), durée de vie résiduelle, L10 contre durées observées
   - choix de la loi par la vraisemblance (AIC, surface de vraisemblance), B10 et B50 avec intervalles, covariables (charge)
3. **Santé et FPT :** détection du premier défaut avec les fausses alarmes annulées, effet du seuil (2 à 5 σ), classement des features, choix du HI, courbes de HI.
4. **Pronostic de la RUL :** courbes de RUL de chaque roulement de test pour tous les modèles (avec l'intervalle du Wiener), prédiction à la troncature, hyperparamètres retenus.
5. **Comparaison des modèles :** les deux protocoles, RMSE par roulement, couverture de l'intervalle du Wiener, fonction de score, coût de calcul, tableau récapitulatif, explicabilité (XAI), covariables dans les modèles ML.
6. **Décision et économie :** curseurs pour changer les coûts, le délai d'intervention et le seuil de décision ; coût par heure de chaque stratégie, âge optimal T*, effet du seuil (tout est recalculé en direct).
7. **Tests et validation :** les six types de tests, données simulées (Weibull et Wiener), tests statistiques, liste des tests automatiques.

## 5. Figures du rapport (`figures/`)

| Figure | Contenu | Sprint |
|---|---|---|
| fig01_signaux_bruts | amplitude brute sur toute la vie des 6 roulements d'apprentissage | 1 |
| fig02_pretraitement | signal brut et filtré, début et fin de vie | 1 |
| fig03_durees_de_vie | durées de vie observées, troncatures et L10 | 1 |
| fig04_fiabilite | Kaplan-Meier et Weibull, λ(t), durée de vie résiduelle | 1 |
| fig05_normalisation | RMS avant et après normalisation par l'état sain | 1 |
| fig06_detection_fpt | règle des 3 σ, fausses alarmes annulées | 2 |
| fig07_classement_features | les 15 meilleures features selon les trois critères | 2 |
| fig08_choix_hi | score du HI en fonction du nombre *k* de features | 2 |
| fig09_hi_apprentissage | HI des roulements d'apprentissage et seuil D | 2 |
| fig10_hi_test | HI des 11 roulements de test, FPT et troncature | 2 |
| fig11_trajectoires_rul | RUL prédite contre RUL réelle, tous les modèles | 3 |
| fig12_wiener_incertitude | intervalle de prédiction à 90 % du Wiener | 3, 4 |
| fig13_comparaison_modeles | RMSE, MAE et score PHM avec intervalles bootstrap | 4 |
| fig14_erreurs_troncature | %Er à la troncature (protocole du challenge) | 4 |
| fig15_fonction_score | fonction de score asymétrique du challenge | 4 |
| fig16_apprentissage_deep | courbes d'apprentissage du LSTM et du CNN 1D | 3 |
| fig17_rmse_par_roulement | RMSE par roulement et par modèle | 4 |
| fig18_vraisemblance | 4 lois face à Kaplan-Meier, surface de vraisemblance de Weibull | 1 |
| fig19_covariables | fiabilité selon la charge (Weibull à temps accéléré), B10 par condition | 1 |
| fig20_donnees_simulees | estimation de β selon le nombre de roulements, couverture du Wiener | 4 |
| fig21_xai | importance de chaque entrée par permutation (SVR, GPR, forêt) | 4 |
| fig22_economie | âge optimal T* et coût par heure de chaque stratégie | 4 |
| fig23_seuil_decision | effet du seuil de décision sur le coût | 4 |

## 6. Rapport (`rapport/`)

- `Rapport_Pronostic_Roulements.pdf` : le rapport final (30 pages, en français), sur le modèle des rapports de projet de l'ENSIT.
- `main.tex` et `chapitres/` : les sources LaTeX. Pour les modifier sur Overleaf, importer le fichier `Rapport_Overleaf.zip` (Nouveau projet > Importer un projet) ; le compilateur XeLaTeX est choisi par `.latexmkrc` (sinon : Menu > Compilateur > XeLaTeX ; pdfLaTeX marche aussi). Les noms de l'équipe se changent dans `chapitres/00_titre.tex`. Le mode d'emploi est dans `rapport/LISEZMOI.md`.
- `references.bib` : les références au format IEEE, à importer dans Zotero (Fichier > Importer).

## Références

- P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Chebel-Morello, N. Zerhouni, C. Varnier, *PRONOSTIA: An experimental platform for bearings accelerated degradation tests*, IEEE International Conference on Prognostics and Health Management (PHM), 2012
- FEMTO-ST, *IEEE PHM 2012 Prognostic Challenge: Outline, Experiments, Scoring of Results, Winners*, 2012
- H. Dhungana, T. Rykkje, A. S. Lundervold, *Bearing Prognostics Using the PRONOSTIA Data: A Comparative Study*, IEEE Access, vol. 13, 2025, doi:10.1109/ACCESS.2025.3551772
- J. Coble, J. W. Hines, *Identifying optimal prognostic parameters from data: a genetic algorithms approach*, Annual Conference of the PHM Society, 2009
- X.-S. Si, W. Wang, C.-H. Hu, D.-H. Zhou, *Remaining useful life estimation – A review on the statistical data driven approaches*, European Journal of Operational Research, vol. 213, no. 1, 2011
- Y. Lei, N. Li, L. Guo, N. Li, T. Yan, J. Lin, *Machinery health prognostics: A systematic review from data acquisition to RUL prediction*, Mechanical Systems and Signal Processing, vol. 104, 2018
- E. L. Kaplan, P. Meier, *Nonparametric estimation from incomplete observations*, Journal of the American Statistical Association, vol. 53, 1958
- ISO 281:2007, *Roulements — Charges dynamiques de base et durée nominale*
- F. Ben Hmida, support du cours *Fiabilité et maintenance* et *Guide de formation sur Statistics and ML et PdM Toolboxes*, ENSIT

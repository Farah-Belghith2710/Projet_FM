"""ÉTAPE 3 — Figures du rapport (PNG 300 dpi, dossier figures/), à partir de results/.

    python 3_figures_rapport.py          (moins d'une minute)

Toutes les figures utilisent la même palette : une couleur fixe par modèle,
la RUL réelle en noir, et des gris discrets pour les axes et la grille.
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                      # pas de fenêtre : on enregistre seulement les images
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src import config  # noqa: E402

# ---------------------------------------------------------------- style commun
ENCRE, ENCRE_2, DISCRET = "#0b0b0b", "#52514e", "#898781"
GRILLE, AXE, FOND = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COULEURS_MODELES = {
    "Fiabilité (B10)": DISCRET, "Wiener": "#2a78d6", "SVR": "#eb6834", "GPR": "#1baf7a",
    "Random Forest": "#eda100", "LSTM": "#e87ba4", "CNN 1D": "#008300",
}
CATEGORIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
COULEURS_CONDITIONS = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}   # charge croissante -> bleu plus foncé
SAIN, DEGRADE = "#0ca30c", "#ec835a"

plt.rcParams.update({
    "figure.facecolor": FOND, "axes.facecolor": FOND, "savefig.facecolor": FOND,
    "axes.edgecolor": AXE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.6,
    "text.color": ENCRE, "axes.labelcolor": ENCRE_2, "xtick.color": ENCRE_2, "ytick.color": ENCRE_2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 8.5, "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8,
})

R = config.DOSSIER_RESULTATS
with open(R / "donnees.json", encoding="utf-8") as f:
    DONNEES = json.load(f)
with open(R / "sante.json", encoding="utf-8") as f:
    SANTE = json.load(f)
with open(R / "fiabilite.json", encoding="utf-8") as f:
    FIAB = json.load(f)
with open(R / "rul.json", encoding="utf-8") as f:
    RUL = json.load(f)
with open(config.DOSSIER_TRAITE / "signaux.json", encoding="utf-8") as f:
    SIGNAUX = json.load(f)
SERIES = pd.read_csv(R / "series.csv")
ROULEMENTS = pd.DataFrame(DONNEES["roulements"])


def enregistrer(fig, nom):
    config.DOSSIER_FIGURES.mkdir(exist_ok=True)
    fig.savefig(config.DOSSIER_FIGURES / nom, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", nom)


def serie(roulement):
    return SERIES[SERIES["roulement"] == roulement]


# ---------------------------------------------------------------- Sprint 1 : données
def fig_signaux_bruts():
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), sharey=True)
    for ax, roulement in zip(axes.ravel(), config.APPRENTISSAGE):
        s = serie(roulement)
        ax.fill_between(s["temps_min"], s["h_min_brut"], s["h_max_brut"], color="#2a78d6", lw=0, alpha=0.85,
                        label="horizontal")
        ax.fill_between(s["temps_min"], s["v_min_brut"], s["v_max_brut"], color="#eb6834", lw=0, alpha=0.55,
                        label="vertical")
        ax.axhline(20, color=DISCRET, lw=0.8)
        ax.axhline(-20, color=DISCRET, lw=0.8)
        ax.set_title(f"{roulement} (condition {roulement[7]})")
        ax.set_ylim(-52, 52)
    axes[0, 0].set_ylabel("accélération (g)")
    axes[1, 0].set_ylabel("accélération (g)")
    for ax in axes[1]:
        ax.set_xlabel("temps (min)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Amplitude brute des vibrations sur toute la vie des 6 roulements d'apprentissage "
                 "(trait gris : ±20 g)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    enregistrer(fig, "fig01_signaux_bruts.png")


def fig_pretraitement():
    roulement = "Bearing1_1"
    s = SIGNAUX[roulement]
    t = np.arange(config.N_POINTS) / config.FS * 1000            # millisecondes
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)
    for ax, k, nom in zip(axes, [0, 2], ["début de vie", "fin de vie"]):
        ax.plot(t, s["brut_h"][k], color=AXE, lw=0.7, label="brut")
        ax.plot(t, s["filtre_h"][k], color="#2a78d6", lw=0.7, label="filtré (10 Hz – 10 kHz)")
        ax.set_title(f"{roulement}, {nom} (t = {s['instants_s'][k] / 60:.0f} min)")
        ax.set_xlabel("temps (ms)")
    axes[0].set_ylabel("accélération horizontale (g)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    enregistrer(fig, "fig02_pretraitement.png")


def fig_durees_de_vie():
    tab = ROULEMENTS.sort_values(["condition", "duree_vie_min"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(tab))
    couleurs = [COULEURS_CONDITIONS[c] for c in tab["condition"]]
    ax.barh(y, tab["duree_vie_min"], height=0.6, color=couleurs)
    for i, (_, ligne) in enumerate(tab.iterrows()):
        if ligne["ensemble"] == "test":
            ax.plot(ligne["troncature"] * config.PAS_TEMPS / 60, i, marker="|", color=ENCRE, ms=12, mew=2)
        ax.text(ligne["duree_vie_min"] + 4, i, f"{ligne['duree_vie_min']:.0f} min", va="center",
                color=ENCRE_2, fontsize=7.5)
    for l10 in FIAB["l10"]:
        lignes = [i for i, c in enumerate(tab["condition"]) if c == l10["condition"]]
        ax.plot([l10["l10_min"]] * 2, [min(lignes) - 0.4, max(lignes) + 0.4], color=ENCRE, lw=1.2, ls="--")
        ax.text(l10["l10_min"] + 4, max(lignes) + 0.15, f"L10 = {l10['l10_min']:.0f} min", fontsize=7.5)
    etiquettes = [f"{r} ({'app.' if e == 'apprentissage' else 'test'})" for r, e in zip(tab["roulement"], tab["ensemble"])]
    ax.set_yticks(y, etiquettes)
    ax.set_xlabel("durée de vie (min)")
    ax.set_xlim(0, 640)
    ax.grid(axis="y", visible=False)
    legende = [Patch(color=couleur, label=f"condition {c} ({config.CONDITIONS[c]['charge_N']} N)")
               for c, couleur in COULEURS_CONDITIONS.items()]
    legende.append(Line2D([], [], marker="|", color=ENCRE, ls="none", ms=10, mew=2, label="troncature (challenge)"))
    legende.append(Line2D([], [], color=ENCRE, ls="--", label="L10 du catalogue"))
    ax.legend(handles=legende, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncols=5)
    ax.set_title("Durées de vie observées et durée de vie nominale L10 des catalogues")
    fig.tight_layout()
    enregistrer(fig, "fig03_durees_de_vie.png")


def fig_fiabilite():
    km, c, w = FIAB["kaplan_meier"], FIAB["courbes"], FIAB["weibull"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    ax.fill_between(km["t"], km["bas"], km["haut"], step="post", color="#2a78d6", alpha=0.12, lw=0)
    ax.step(km["t"], km["survie"], where="post", color="#2a78d6", label="Kaplan-Meier (IC 95 %)")
    ax.plot(c["t_min"], c["fiabilite"], color=ENCRE, lw=1.2, label=f"Weibull β = {w['beta']:.2f}, η = {w['eta_min']:.0f} min")
    censures = [d["duree_min"] for d in FIAB["donnees"] if d["evenement"] == 0]
    ax.plot(censures, np.interp(censures, km["t"], km["survie"]), "|", color=ENCRE_2, ms=8, label="censure (test)")
    ax.set_xlabel("temps (min)")
    ax.set_ylabel("fiabilité R(t)")
    ax.set_title("Fiabilité de la population")
    ax.legend(loc="lower left")
    axes[1].plot(c["t_min"], c["taux_par_heure"], color="#2a78d6")
    axes[1].set_xlabel("temps (min)")
    axes[1].set_ylabel("λ(t) (défaillances par heure)")
    axes[1].set_title("Taux de défaillance (β > 1 : usure)")
    axes[2].plot(c["t_min"], c["mrl_min"], color="#2a78d6", label="moyenne (MRL)")
    axes[2].plot(c["t_min"], c["b10_residuel_min"], color="#eb6834", label="B10 conditionnel (90 %)")
    axes[2].set_xlabel("âge du roulement (min)")
    axes[2].set_ylabel("durée de vie résiduelle (min)")
    axes[2].set_title("Durée de vie résiduelle d'un roulement sain")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    enregistrer(fig, "fig04_fiabilite.png")


# ---------------------------------------------------------------- Sprint 2 : santé
def fig_normalisation():
    exemples = ["Bearing1_1", "Bearing2_1", "Bearing3_2"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for roulement, couleur in zip(exemples, CATEGORIES):
        s = serie(roulement)
        axes[0].plot(s["temps_min"], s["h_rms_brut"], color=couleur, lw=0.8, label=roulement)
        axes[1].plot(s["temps_min"], s["h_rms_norm"], color=couleur, lw=0.8, label=roulement)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMS horizontal (g)")
    axes[0].set_title("Avant : niveaux différents selon la condition")
    axes[1].set_ylabel("log(RMS / RMS à l'état sain)")
    axes[1].set_title("Après : 0 = état sain pour tous les roulements")
    for ax in axes:
        ax.set_xlabel("temps (min)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    enregistrer(fig, "fig05_normalisation.png")


def fig_fpt():
    exemples = ["Bearing2_3", "Bearing1_3"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    for ax, roulement in zip(axes, exemples):
        s, info = serie(roulement), SANTE["roulements"][roulement]
        for debut, fin in info["episodes"]:
            annulee = fin < s["index"].max()
            ax.axvspan(debut * 10 / 60, fin * 10 / 60, color=AXE if annulee else DEGRADE, alpha=0.25, lw=0)
        ax.plot(s["temps_min"], s["h_rms_norm"], color="#2a78d6", lw=0.8, label="RMS horizontal (lissé)")
        ax.plot(s["temps_min"], s["v_rms_norm"], color="#eb6834", lw=0.8, label="RMS vertical (lissé)")
        ax.axhline(info["seuil_h"], color="#2a78d6", lw=0.8, ls="--")
        ax.axhline(info["seuil_v"], color="#eb6834", lw=0.8, ls="--")
        ax.axvline(info["fpt_min"], color=ENCRE, lw=1.2)
        ax.text(info["fpt_min"], ax.get_ylim()[1] * 0.92, f"  FPT = {info['fpt_min']:.0f} min", fontsize=8)
        tr = config.TEST[roulement]["troncature"] * 10 / 60
        ax.axvline(tr, color=ENCRE_2, lw=1, ls=":")
        ax.set_title(f"{roulement} : {len(info['episodes']) - 1} fausse(s) alarme(s) annulée(s) (gris), "
                     "défaut confirmé (orange), troncature (pointillés)")
        ax.set_ylabel("log(RMS / sain)")
    axes[1].set_xlabel("temps (min)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    enregistrer(fig, "fig06_detection_fpt.png")


def fig_classement():
    tab = pd.DataFrame(SANTE["classement"]).head(15).iloc[::-1]
    retenues = set(SANTE["selection"])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    couleurs = ["#2a78d6" if f in retenues else AXE for f in tab["feature"]]
    ax.barh(tab["feature"], tab["score"], height=0.6, color=couleurs)
    ax.plot(tab["monotonie"], tab["feature"], "o", color=ENCRE, ms=3, label="monotonie")
    ax.plot(tab["correlation"], tab["feature"], "s", color=ENCRE_2, ms=3, label="corrélation")
    ax.plot(tab["pronosticabilite"], tab["feature"], "^", color=DISCRET, ms=3.5, label="pronosticabilité")
    ax.set_xlim(0, 1)
    ax.set_xlabel("critère (0 à 1) — barre = moyenne des trois")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Classement des 15 meilleures features (en bleu : les 8 entrées des modèles ML et deep)")
    fig.tight_layout()
    enregistrer(fig, "fig07_classement_features.png")


def fig_choix_hi():
    essais = pd.DataFrame(SANTE["choix_hi"])
    k_retenu = len(SANTE["colonnes_hi"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(essais["k"], essais["monotonie"], "o-", color=ENCRE, lw=1, ms=3.5, label="monotonie")
    ax.plot(essais["k"], essais["correlation"], "s-", color=ENCRE_2, lw=1, ms=3.5, label="corrélation")
    ax.plot(essais["k"], essais["pronosticabilite"], "^-", color=DISCRET, lw=1, ms=4, label="pronosticabilité")
    ax.plot(essais["k"], essais["score"], "o-", color="#2a78d6", lw=2.2, ms=5, label="score (moyenne)")
    ax.axvline(k_retenu, color=ENCRE, lw=0.8, ls=":")
    ax.text(k_retenu + 0.15, 0.04, f"k retenu = {k_retenu}", fontsize=8)
    ax.set_xticks(essais["k"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("nombre k de features moyennées dans le HI")
    ax.set_ylabel("critère (0 à 1)")
    ax.legend(loc="lower right", ncols=2, fontsize=7.5)
    ax.set_title("Choix du HI : moyenne des k meilleures features (roulements d'apprentissage)")
    fig.tight_layout()
    enregistrer(fig, "fig08_choix_hi.png")


def fig_hi_apprentissage():
    seuil = RUL["infos_modeles"]["Wiener"]["parametres"]["seuil"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for roulement, couleur in zip(config.APPRENTISSAGE, CATEGORIES):
        s = serie(roulement)
        axes[0].plot(s["temps_min"], s["hi"], color=couleur, lw=0.8, label=roulement)
        phase = s[s["index"] >= s["fpt"].iloc[0]]
        pourcentage = 100 * (phase["index"] - phase["index"].iloc[0]) / (phase["index"].iloc[-1] - phase["index"].iloc[0])
        axes[1].plot(pourcentage, phase["hi"], color=couleur, lw=0.9)
    for ax in axes:
        ax.axhline(seuil, color=ENCRE, lw=1, ls="--")
    axes[1].text(2, seuil + 0.1, f"seuil de défaillance D = {seuil:.2f}", fontsize=8)
    axes[0].set_xlabel("temps (min)")
    axes[1].set_xlabel("avancement de la phase de dégradation (%)")
    axes[0].set_ylabel("HI")
    axes[0].set_title("HI sur toute la vie")
    axes[1].set_title("HI après le FPT")
    poignees, noms = axes[0].get_legend_handles_labels()
    fig.legend(poignees, noms, loc="lower center", ncols=6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    enregistrer(fig, "fig09_hi_apprentissage.png")


def fig_hi_test():
    seuil = RUL["infos_modeles"]["Wiener"]["parametres"]["seuil"]
    fig, axes = plt.subplots(4, 3, figsize=(10, 8.5))
    for ax, roulement in zip(axes.ravel(), config.TEST):
        s, info = serie(roulement), SANTE["roulements"][roulement]
        ax.plot(s["temps_min"], s["hi"], color="#2a78d6", lw=0.7)
        ax.axhline(seuil, color=ENCRE, lw=0.8, ls="--")
        ax.axvline(info["fpt_min"], color=DEGRADE, lw=1.2)
        ax.axvline(config.TEST[roulement]["troncature"] * 10 / 60, color=ENCRE_2, lw=1, ls=":")
        ax.set_title(roulement, fontsize=9)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0, 0.6, "— HI\n– – seuil D\n| FPT (orange)\n: troncature du challenge", fontsize=9,
                          color=ENCRE_2, va="center")
    fig.suptitle("Indicateur de santé des 11 roulements de test", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    enregistrer(fig, "fig10_hi_test.png")


# ---------------------------------------------------------------- Sprints 3 et 4 : RUL
def fig_trajectoires():
    exemples = ["Bearing1_3", "Bearing1_4", "Bearing3_3", "Bearing1_7"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, roulement in zip(axes.ravel(), exemples):
        c = RUL["courbes"][roulement]
        depuis = np.array(c["temps_min"]) - c["temps_min"][0]
        for nom in RUL["modeles"]:
            ax.plot(depuis, c["predictions"][nom], color=COULEURS_MODELES[nom], lw=1.1, label=nom,
                    ls="--" if nom == "Fiabilité (B10)" else "-")
        ax.plot(depuis, c["vrai_min"], color=ENCRE, lw=1.8, label="RUL réelle")
        haut = max(c["vrai_min"]) * 2.2
        ax.set_ylim(0, haut)
        ax.set_title(f"{roulement} (phase de dégradation : {max(c['vrai_min']):.0f} min)")
        ax.set_xlabel("temps depuis le FPT (min)")
        ax.set_ylabel("RUL (min)")
    axes[0, 0].legend(loc="upper right", ncols=2, fontsize=7)
    fig.tight_layout()
    enregistrer(fig, "fig11_trajectoires_rul.png")


def fig_wiener():
    exemples = ["Bearing1_3", "Bearing3_3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, roulement in zip(axes, exemples):
        c = RUL["courbes"][roulement]
        depuis = np.array(c["temps_min"]) - c["temps_min"][0]
        ax.fill_between(depuis, c["wiener_bas"], c["wiener_haut"], color="#2a78d6", alpha=0.15, lw=0,
                        label="intervalle de prédiction 90 %")
        ax.plot(depuis, c["predictions"]["Wiener"], color="#2a78d6", label="Wiener (médiane)")
        ax.plot(depuis, c["vrai_min"], color=ENCRE, lw=1.8, label="RUL réelle")
        ax.set_ylim(0, max(c["vrai_min"]) * 2.2)
        ax.set_title(roulement)
        ax.set_xlabel("temps depuis le FPT (min)")
        ax.set_ylabel("RUL (min)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    enregistrer(fig, "fig12_wiener_incertitude.png")


def fig_comparaison():
    tab = pd.DataFrame(RUL["metriques"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, colonne, titre, ic in [(axes[0], "rmse_min", "RMSE (min) — plus bas = mieux", "rmse_ic95"),
                                   (axes[1], "mae_min", "MAE (min) — plus bas = mieux", None),
                                   (axes[2], "score_phase", "Score PHM — plus haut = mieux", "score_phase_ic95")]:
        t = tab.sort_values(colonne, ascending=(colonne == "score_phase"))
        y = np.arange(len(t))
        ax.barh(y, t[colonne], height=0.55, color=[COULEURS_MODELES[m] for m in t["modele"]])
        if ic:
            bas = [v[0] for v in t[ic]]
            haut = [v[1] for v in t[ic]]
            ax.errorbar(t[colonne], y, xerr=[t[colonne] - bas, np.array(haut) - t[colonne]], fmt="none",
                        ecolor=ENCRE_2, elinewidth=0.9, capsize=2)
        ax.set_yticks(y, t["modele"])
        ax.grid(axis="y", visible=False)
        ax.set_title(titre)
    fig.suptitle("Comparaison sur la phase de dégradation des 11 roulements de test "
                 "(barres d'erreur : IC 95 % par bootstrap)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    enregistrer(fig, "fig13_comparaison_modeles.png")


def fig_troncature():
    tab = RUL["troncature"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(tab))
    decalages = np.linspace(-0.3, 0.3, len(RUL["modeles"]))
    for d, nom in zip(decalages, RUL["modeles"]):
        erreurs = np.clip([ligne[nom]["erreur_pct"] for ligne in tab], -300, 100)
        ax.plot(x + d, erreurs, "o", color=COULEURS_MODELES[nom], ms=4, label=nom)
    for i, ligne in enumerate(tab):
        ax.axvspan(i - 0.45, i + 0.45, color=SAIN if ligne["etat"] == "sain" else DEGRADE, alpha=0.08, lw=0)
    ax.axhline(0, color=ENCRE, lw=0.8)
    ax.set_xticks(x, [f"{ligne['roulement']}\n({ligne['etat']})" for ligne in tab], fontsize=7)
    ax.set_ylabel("%Er (bornée à -300 %)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Erreur à l'instant de troncature (protocole du challenge) : %Er < 0 = prédiction trop tardive")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=len(RUL["modeles"]), fontsize=7)
    fig.tight_layout()
    enregistrer(fig, "fig14_erreurs_troncature.png")


def fig_fonction_score():
    er = np.linspace(-50, 50, 400)
    a = np.where(er <= 0, np.exp(-np.log(0.5) * er / 5), np.exp(np.log(0.5) * er / 20))
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.plot(er, a, color="#2a78d6")
    ax.axvline(0, color=AXE, lw=0.8)
    ax.text(-48, 0.85, "prédiction tardive\n(dangereuse)", fontsize=8, color=ENCRE_2)
    ax.text(18, 0.85, "prédiction précoce\n(prudente)", fontsize=8, color=ENCRE_2)
    ax.set_xlabel("%Er = 100 × (RUL réelle − RUL prédite) / RUL réelle")
    ax.set_ylabel("score A")
    ax.set_title("Fonction de score du challenge IEEE PHM 2012")
    fig.tight_layout()
    enregistrer(fig, "fig15_fonction_score.png")


def fig_apprentissage_deep():
    noms = [n for n in ("LSTM", "CNN 1D") if n in RUL["infos_modeles"]]
    if not noms:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for nom in noms:
        courbe = RUL["infos_modeles"][nom]["courbe_apprentissage"]
        ax.plot(np.arange(1, len(courbe) + 1), courbe, color=COULEURS_MODELES[nom], label=nom)
    ax.set_xlabel("époque")
    ax.set_ylabel("erreur absolue moyenne (log RUL)")
    ax.set_title("Apprentissage des réseaux de neurones")
    ax.legend()
    fig.tight_layout()
    enregistrer(fig, "fig16_apprentissage_deep.png")


def fig_rmse_par_roulement():
    noms = RUL["modeles"]
    valeurs = np.array([RUL["par_roulement"][n]["rmse_min"] for n in noms])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    image = ax.imshow(np.log10(valeurs + 1), cmap="Blues", aspect="auto")
    for i in range(len(noms)):
        for j in range(len(RUL["roulements"])):
            v = valeurs[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if np.log10(v + 1) > 1.6 else ENCRE)
    ax.set_xticks(range(len(RUL["roulements"])), RUL["roulements"], fontsize=7, rotation=30)
    ax.set_yticks(range(len(noms)), noms)
    ax.grid(False)
    ax.set_title("RMSE (min) par roulement de test et par modèle (couleur : échelle logarithmique)")
    fig.colorbar(image, ax=ax, fraction=0.02, label="log10(RMSE + 1)")
    fig.tight_layout()
    enregistrer(fig, "fig17_rmse_par_roulement.png")


# ---------------------------------------------------------------- étape 4 : analyses complémentaires
FICHIER_COMPLEMENTS = R / "complements.json"
COMP = json.load(open(FICHIER_COMPLEMENTS, encoding="utf-8")) if FICHIER_COMPLEMENTS.exists() else None
ACCENT = "#B8572F"                                     # terre cuite : la loi ou la stratégie retenue
INDIGO = ["#B8C7E6", "#6F8CC4", "#283F6B"]             # une teinte, du petit au grand échantillon


def fig_vraisemblance():
    v, km = COMP["vraisemblance"], FIAB["kaplan_meier"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.fill_between(km["t"], km["bas"], km["haut"], step="post", color=DISCRET, alpha=0.15, lw=0)
    ax.step(km["t"], km["survie"], where="post", color=ENCRE, lw=1.4, label="Kaplan-Meier (IC 95 %)")
    styles = {"Exponentielle": ("#c3c2b7", "--"), "Log-normale": ("#898781", "-"),
              "Log-logistique": ("#52514e", ":"), "Weibull": (ACCENT, "-")}
    aic = {ligne["loi"]: ligne["aic"] for ligne in v["lois"]}
    for loi, (couleur, trait) in styles.items():
        ax.plot(v["courbes"]["t_min"], v["courbes"][loi], color=couleur, ls=trait, lw=2 if loi == "Weibull" else 1.3,
                label=f"{loi} (AIC {aic[loi]:.1f})")
    ax.set_xlabel("temps (min)")
    ax.set_ylabel("fiabilité R(t)")
    ax.set_title("Quatre lois ajustées par maximum de vraisemblance")
    ax.legend(loc="lower left", fontsize=7.5)
    ax = axes[1]
    s = v["surface"]
    z = np.array(s["log_vraisemblance"])
    niveaux = np.linspace(s["seuil_region_95"] - 12, z.max(), 12)
    remplissage = ax.contourf(s["beta"], s["eta_min"], z, levels=niveaux, cmap="Blues", extend="min")
    ax.contour(s["beta"], s["eta_min"], z, levels=[s["seuil_region_95"]], colors=ENCRE, linestyles="--", linewidths=1.3)
    w = FIAB["weibull"]
    ax.plot(w["beta"], w["eta_min"], "x", color=ACCENT, ms=9, mew=2.2, label="maximum de vraisemblance")
    ax.plot([], [], ls="--", color=ENCRE, label="région de confiance 95 %")
    ax.set_xlabel("forme β")
    ax.set_ylabel("échelle η (min)")
    ax.set_title("Log-vraisemblance de Weibull (avec censure)")
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.colorbar(remplissage, ax=ax, fraction=0.04, label="log-vraisemblance")
    fig.tight_layout()
    enregistrer(fig, "fig18_vraisemblance.png")


def fig_covariables():
    cov = COMP["covariables_fiabilite"]
    charge = next(m for m in cov["modeles"] if m["covariable"].startswith("charge"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    for c, kn in zip((1, 2, 3), ("4,0", "4,2", "5,0")):
        ax.plot(cov["courbes_charge"]["t_min"], cov["courbes_charge"][f"condition {c}"], color=COULEURS_CONDITIONS[c],
                lw=2, label=f"condition {c} ({kn} kN)")
    for d in FIAB["donnees"]:
        if d["evenement"] == 1:
            ax.plot(d["duree_min"], 0.02, "^", color=COULEURS_CONDITIONS[int(d["roulement"][7])], ms=7, clip_on=False)
    ax.set_xlabel("temps (min)  (triangles : défaillances observées)")
    ax.set_ylabel("fiabilité R(t)")
    ax.set_title(f"Weibull à temps accéléré : +1 kN de charge → durée de vie × {charge['facteur_duree_de_vie']:.2f}")
    ax.legend(loc="upper right")
    ax = axes[1]
    b10 = charge["b10_par_condition_min"]
    x = np.arange(3)
    barres = ax.bar(x, [b10[str(c)] for c in (1, 2, 3)], width=0.55, color=[COULEURS_CONDITIONS[c] for c in (1, 2, 3)])
    ax.bar_label(barres, fmt="%.0f min", padding=3, fontsize=8)
    ax.axhline(FIAB["weibull"]["b10_min"], color=ENCRE_2, ls="--", lw=1)
    ax.text(2.45, FIAB["weibull"]["b10_min"] + 4, "B10 sans covariable", ha="right", fontsize=7.5, color=ENCRE_2)
    ax.set_xticks(x, ["condition 1", "condition 2", "condition 3"])
    ax.set_ylabel("B10 (min)")
    ax.set_title("B10 selon la condition opératoire")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(b10.values()) * 1.25)
    fig.tight_layout()
    enregistrer(fig, "fig19_covariables.png")


def fig_donnees_simulees():
    sw, sim = COMP["simulation_weibull"], COMP["simulation_wiener"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    tailles = list(sw["par_taille"])
    boites = ax.boxplot([sw["par_taille"][n]["betas"] for n in tailles], widths=0.5, patch_artist=True,
                        showfliers=False, medianprops=dict(color=ENCRE))
    for boite, couleur in zip(boites["boxes"], INDIGO):
        boite.set(facecolor=couleur, edgecolor=ENCRE_2)
    ax.axhline(sw["beta_vrai"], color=ACCENT, ls="--", lw=1.3, label=f"vrai β = {sw['beta_vrai']:.2f}")
    ax.set_xticks(range(1, len(tailles) + 1), [f"{n} roulements" for n in tailles])
    ax.set_ylabel("β estimé (400 simulations)")
    ax.set_title("Weibull : β retrouvé, mais très incertain avec 17 roulements")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    scenarios = [k for k in sim if k != "seuil_test_observe"]
    noms = scenarios + ["vraies données (11 roulements de test)"]
    valeurs = [100 * sim[s]["couverture_90"] for s in scenarios] + [100 * RUL["couverture_wiener_90"]["moyenne"]]
    y = np.arange(len(noms))[::-1]
    barres = ax.barh(y, valeurs, height=0.55, color=["#6F8CC4"] * len(scenarios) + [ACCENT])
    ax.bar_label(barres, fmt="%.0f %%", padding=3, fontsize=8)
    ax.axvline(90, color=ENCRE, ls="--", lw=1)
    ax.text(89, -0.62, "objectif 90 %", ha="right", fontsize=7.5)
    ax.set_yticks(y, noms)
    ax.set_xlim(0, 108)
    ax.set_xlabel("part des vraies RUL dans l'intervalle à 90 % du Wiener")
    ax.set_title("Wiener : correct si ses hypothèses sont vraies")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    enregistrer(fig, "fig20_donnees_simulees.png")


def fig_xai():
    xai = COMP["explicabilite"]
    modeles = ["SVR", "GPR", "Random Forest"]
    tables = {m: {d["nom"]: d["hausse_mae_min"] for d in xai[m]["importances"]} for m in modeles}
    entrees = sorted(tables["SVR"], key=lambda n: np.mean([tables[m][n] for m in modeles]))
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(entrees))
    for k, m in enumerate(modeles):
        ax.barh(y + (k - 1) * 0.26, [tables[m][n] for n in entrees], height=0.25, color=COULEURS_MODELES[m], label=m)
    ax.axvline(0, color=AXE, lw=0.8)
    ax.set_yticks(y, entrees)
    ax.set_xlabel("hausse de l'erreur MAE (min) quand on mélange cette entrée")
    ax.set_title("Explicabilité : importance de chaque entrée par permutation")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    enregistrer(fig, "fig21_xai.png")


GRIS_STRATEGIES = {"Corrective": "#6B5E55", "Préventive (âge T*)": "#A3968A", "Conditionnelle (dès le FPT)": "#C4B6A6",
                   "Idéale (juste avant la panne)": "#E2D8CB", "Prédictive (Wiener prudent, 5 %)": "#8DB8EE"}


def couleur_strategie(nom):
    return GRIS_STRATEGIES.get(nom) or COULEURS_MODELES[nom.replace("Prédictive (", "").rstrip(")")]


def fig_economie():
    eco = COMP["economie"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    courbe = eco["courbe_age"]
    ax.plot(courbe["age_min"], courbe["cout_par_heure_u"], color=ACCENT, lw=2, label="remplacement à l'âge T")
    corrective = eco["strategies"]["Corrective"]
    c_corr = eco["parametres"]["cc"] / FIAB["weibull"]["mttf_min"] * 60
    ax.axhline(c_corr, color=ENCRE_2, ls=":", lw=1.2, label="tout correctif (Cc / MTTF)")
    k = int(np.argmin(courbe["cout_par_heure_u"]))
    ax.plot(eco["t_etoile_min"], courbe["cout_par_heure_u"][k], "o", color=ENCRE, ms=7)
    ax.annotate(f"T* = {eco['t_etoile_min']:.0f} min", (eco["t_etoile_min"], courbe["cout_par_heure_u"][k]),
                xytext=(10, -16), textcoords="offset points", fontsize=8)
    ax.set_ylim(0, c_corr * 2)
    ax.set_xlabel("âge de remplacement T (min)")
    ax.set_ylabel("coût par heure (u/h)")
    ax.set_title("Préventif : l'âge optimal T* (Cc = 5 Cp)")
    ax.legend(loc="upper right")
    ax = axes[1]
    strategies = sorted(eco["strategies"].items(), key=lambda kv: kv[1]["cout_par_heure_u"], reverse=True)
    y = np.arange(len(strategies))
    barres = ax.barh(y, [b["cout_par_heure_u"] for _, b in strategies], height=0.6,
                     color=[couleur_strategie(n) for n, _ in strategies])
    ax.bar_label(barres, labels=[f"{b['cout_par_heure_u']:.2f} · {b['pannes']} panne(s)" for _, b in strategies],
                 padding=3, fontsize=7.5)
    ax.set_yticks(y, [n for n, _ in strategies], fontsize=7.5)
    ax.set_xlim(0, corrective["cout_par_heure_u"] * 1.45)
    ax.set_xlabel("coût par heure de fonctionnement (u/h)")
    ax.set_title("Les stratégies rejouées sur les 11 roulements de test")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    enregistrer(fig, "fig22_economie.png")


def fig_seuil_decision():
    eco = COMP["economie"]
    table = pd.DataFrame(eco["sensibilite_seuil"])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for colonne in table.columns:
        if colonne == "seuil_min":
            continue
        couleur = COULEURS_MODELES.get(colonne, "#8DB8EE")
        ax.plot(table["seuil_min"], table[colonne], "o-", color=couleur, ms=3.5, lw=1.4, label=colonne)
    conditionnelle = eco["strategies"]["Conditionnelle (dès le FPT)"]["cout_par_heure_u"]
    ax.axhline(conditionnelle, color=ENCRE_2, ls="--", lw=1, label="conditionnelle (dès le FPT)")
    ax.set_xlabel("seuil de décision s (min) : on remplace quand la RUL prédite passe sous s")
    ax.set_ylabel("coût par heure (u/h)")
    ax.set_title("Sensibilité au seuil de décision")
    ax.legend(loc="upper right", ncols=2, fontsize=7)
    fig.tight_layout()
    enregistrer(fig, "fig23_seuil_decision.png")


if __name__ == "__main__":
    print("Figures du rapport :")
    fig_signaux_bruts()
    fig_pretraitement()
    fig_durees_de_vie()
    fig_fiabilite()
    fig_normalisation()
    fig_fpt()
    fig_classement()
    fig_choix_hi()
    fig_hi_apprentissage()
    fig_hi_test()
    fig_trajectoires()
    fig_wiener()
    fig_comparaison()
    fig_troncature()
    fig_fonction_score()
    fig_apprentissage_deep()
    fig_rmse_par_roulement()
    if COMP is not None:                     # figures de l'étape 4 (si elle a été lancée)
        fig_vraisemblance()
        fig_covariables()
        fig_donnees_simulees()
        fig_xai()
        fig_economie()
        fig_seuil_decision()

"""Dessin de la page de garde : un roulement à billes et son signal de vibration (fichier vectoriel PDF).

    python rapport/illustration_couverture.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402

BAGUE, BILLE, SIGNAL, FOND = "#22748F", "#B8572F", "#B8572F", "white"
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.set_aspect("equal")
ax.axis("off")

# roulement : bague extérieure, bague intérieure, 10 billes, arbre
ax.add_patch(Wedge((0, 0), 1.50, 0, 360, width=0.28, color=BAGUE))
ax.add_patch(Wedge((0, 0), 0.82, 0, 360, width=0.24, color=BAGUE, alpha=0.85))
ax.add_patch(Circle((0, 0), 0.58, color="#E6D9C8"))
for k in range(10):
    angle = 2 * np.pi * k / 10
    ax.add_patch(Circle((1.02 * np.cos(angle), 1.02 * np.sin(angle)), 0.17, color=BILLE))
ax.add_patch(Circle((1.02, 0), 0.17, fill=False, ec="white", lw=1.5, ls="--"))   # la bille qui s'use

# signal de vibration : calme, puis chocs qui grandissent (dégradation)
t = np.linspace(0, 1, 1400)
rng = np.random.default_rng(3)
enveloppe = 0.06 + 0.55 * t ** 4
signal = enveloppe * np.sin(2 * np.pi * 70 * t) + 0.04 * rng.normal(size=t.size)
ax.plot(1.9 + 3.6 * t, signal, color=SIGNAL, lw=0.8)
ax.plot([1.9, 5.5], [-0.75, -0.75], color="#6B5E55", lw=0.8)
ax.annotate("", xy=(5.6, -0.75), xytext=(5.4, -0.75), arrowprops=dict(arrowstyle="->", color="#6B5E55", lw=0.8))
ax.text(5.55, -0.92, "temps", ha="right", va="top", fontsize=8, color="#6B5E55")
ax.set_xlim(-1.6, 5.7)
ax.set_ylim(-1.6, 1.6)
sortie = Path(__file__).resolve().parent / "figures" / "illustration_couverture.pdf"
fig.savefig(sortie, bbox_inches="tight", facecolor=FOND)
print("->", sortie)

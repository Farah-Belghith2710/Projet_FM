# Rapport LaTeX — Pronostic de défaillance de roulements

Rapport final du projet (30 pages, en français). Le PDF compilé est `Rapport_Pronostic_Roulements.pdf`.

## Compiler sur Overleaf

1. Overleaf > **Nouveau projet** > **Importer un projet** > choisir `Rapport_Overleaf.zip`.
2. **Menu** > **Compilateur** : choisir **XeLaTeX** (le fichier `.latexmkrc` le demande déjà).
   pdfLaTeX marche aussi : le résumé arabe est alors inséré comme image (`figures/resume_arabe.pdf`).
3. Cliquer sur **Recompiler** : on obtient le même PDF de 30 pages.

## Ce qu'on modifie le plus souvent

- **Noms de l'équipe** : `chapitres/00_titre.tex`, ligne `\newcommand{\equipe}{...}`.
  Pour plusieurs noms : `\newcommand{\equipe}{Farah Belghith \\ Prénom Nom \\ Prénom Nom}`.
- **Texte** : un fichier par chapitre dans `chapitres/`, dans l'ordre de `main.tex`.
- **Figures** : dossier `figures/` (elles viennent de `3_figures_rapport.py` dans le projet Python).

## Bibliographie et Zotero

- `references.bib` contient les 25 références, affichées au format IEEE (`\bibliographystyle{ieeetr}`).
- Import dans Zotero : **Fichier** > **Importer…** > `references.bib`.
- Pour ajouter une référence : le connecteur Zotero du navigateur l'enregistre depuis IEEE Xplore ou
  Google Scholar ; ensuite clic droit sur la collection > **Exporter la collection…** > **BibTeX**,
  et remplacer `references.bib`. Dans le texte : `\cite{cle}`.

## Compiler sur son ordinateur

- `latexmk main.tex` (XeLaTeX grâce à `.latexmkrc`),
- ou à la main : `xelatex main`, `bibtex main`, puis `xelatex main` deux fois.

Polices : TeX Gyre Termes pour le texte (fournie avec toutes les distributions LaTeX) et Amiri pour
l'arabe (dossier `polices/`, licence libre OFL).

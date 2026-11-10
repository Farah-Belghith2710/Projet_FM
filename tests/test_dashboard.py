"""Chaque page du dashboard s'exécute sans erreur avec les résultats de results/ (comme dans Streamlit)."""
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
PAGES = sorted((DASHBOARD / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_sans_erreur(page):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=90).run()   # l'appli complète (navigation)
    app.switch_page(f"pages/{page.name}").run()
    assert not app.exception

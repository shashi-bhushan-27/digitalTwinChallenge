"""Headless dashboard checks (Streamlit AppTest). Skipped when the demo cache has not been built."""

import re

import pytest

from dashboard import service

pytestmark = pytest.mark.skipif(bool(service.missing_cache()), reason="demo cache missing: run `uv run python -m dashboard.prepare`")

RAW_FEATURE = re.compile(r"\b(cgm|wear|meal|ehr|tod)_[a-z0-9_]+\b")


def _texts(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [e.value for e in at.error] + [w.value for w in at.warning] \
        + [s.value for s in at.success] + [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    from common.config import REPO_ROOT

    at = AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_banner_and_no_raw_feature_names(app):
    text = _texts(app)
    assert "Synthetic patient. Decision support only." in text
    assert not RAW_FEATURE.search(text), RAW_FEATURE.search(text).group(0)


def test_every_scene_renders(app):
    for title in app.selectbox(key="scene").options:
        app.selectbox(key="scene").set_value(title).run()
        assert not app.exception, (title, [e.value for e in app.exception])
        assert not RAW_FEATURE.search(_texts(app))


def test_whatif_interventions_run(app):
    app.selectbox(key="scene").set_value(app.selectbox(key="scene").options[0]).run()
    run = next(b for b in app.button if b.label == "Simulate on the twin")
    run.click().run()
    assert not app.exception, [e.value for e in app.exception]
    text = _texts(app)
    assert "Simulation for discussion with the patient, not a prescription." in text


def test_time_controls(app):
    for label in ("+1 h", "−1 h", "+15 min"):
        next(b for b in app.button if b.label == label).click().run()
        assert not app.exception

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_all_dashboard_sections_and_recipe_change():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "dashboard.py"), default_timeout=20).run()
    assert not app.exception
    for section in [
        "Tool matching",
        "Reliability",
        "Maintenance forecast",
        "Recipe optimization",
        "Recipe sandbox",
    ]:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception, section
    before = app.metric[0].value
    app.slider[0].set_value(850).run()
    assert not app.exception
    assert app.metric[0].value != before
    app.sidebar.radio[0].set_value("Process monitoring").run()
    next(widget for widget in app.selectbox if widget.label == "Injected scenario").set_value(
        "sensor_bias"
    ).run()
    assert not app.exception

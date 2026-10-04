"""Flask dashboard for backtest results and upcoming fixtures.

Run with `flask --app valuemodel.web run`. The dashboard works from the local
database and cached files and never contacts the data source itself. The
database is created empty on the first visit if no backtest has been saved.
"""

from flask import Flask

from valuemodel.config import Settings, load_settings
from valuemodel.web.routes import pages, register_filters


def create_app(settings: Settings | None = None) -> Flask:
    app = Flask(__name__)
    app.config["SETTINGS"] = settings or load_settings()
    register_filters(app)
    app.register_blueprint(pages)
    return app

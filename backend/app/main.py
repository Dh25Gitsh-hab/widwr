"""
Application entry point.

WHY a factory (create_app)
    Production calls ``create_app()`` with no arguments: settings come from
    the environment and data is loaded at start-up. Tests call it with a small
    synthetic DataStore, so they run in milliseconds without files or network.

START-UP SEQUENCE
    1. read settings (config.py)
    2. load + clean the data once (data_store.py) and keep it in app.state
    3. load the trained ML models (prediction_service.py) - optional
    4. if loading fails (no snapshot and Elexon unreachable, or no trained
       models) the server still starts: /api/health reports the reason and
       only the affected endpoints return 503, instead of the process crashing.

Run locally (from the backend folder):
    uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes import analytics, forecasts, health, ml
from app.config import Settings, get_settings
from app.services.data_store import DataStore, load_store
from app.services.elexon_client import ElexonAPIError
from app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

API_DESCRIPTION = """
Decision support for a UK day-ahead power-buying desk.

* **Q1 - what wind forecast did we have at decision time?** `/api/forecasts/day-ahead`
* **Q2 - how accurate was it?** `/api/analytics/*`
* **Extension - ML:** two separate components, `/api/ml/corrected-forecast` (MW) and
  `/api/ml/shortfall-risk` (probability of a >= 1 GW shortfall); `/api/ml/models` lists the
  served models (approved defaults and comparison models)

Data: Elexon BMRS **WINDFOR** (wind forecasts) and **FUELHH** (actual wind
generation), national level, for purchase decisions made in January 2026.
All times are UTC; error = forecast - actual (MW), positive = over-forecast.
"""


def create_app(
    settings: Settings | None = None,
    store: DataStore | None = None,
    predictor: PredictionService | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.load_error = None
        if store is not None:
            app.state.store = store
        else:
            try:
                app.state.store = load_store(settings)
            except (ElexonAPIError, ValueError, OSError) as exc:
                log.error("Could not load data: %s", exc)
                app.state.store = None
                app.state.load_error = str(exc)

        app.state.ml_error = None
        if predictor is not None:
            app.state.predictor = predictor
        else:
            try:
                app.state.predictor = PredictionService.load(
                    settings.ml_dir, settings.ml_default_regressor, settings.ml_default_classifier
                )
                log.info("ML models loaded from %s", settings.ml_dir)
            except (OSError, ValueError, KeyError) as exc:
                log.warning("ML models not available (%s) - see backend/ml/README.md", exc)
                app.state.predictor = None
                app.state.ml_error = (
                    f"{exc} - the artifacts are built by `python -m ml.build_dataset`, `python -m ml.train` "
                    "and `python -m ml.holdout.predict`"
                )
        yield

    app = FastAPI(
        title="Wind Forecast Intelligence API",
        version="1.0.0",
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )
    # Endpoints read settings through Depends(get_settings); make them see the same object.
    app.dependency_overrides[get_settings] = lambda: settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    for module in (health, forecasts, analytics, ml):
        app.include_router(module.router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()

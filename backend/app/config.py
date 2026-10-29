"""
Central configuration.

WHY
    Every value that can differ between environments (my laptop, a reviewer's
    machine, Docker, a cloud host) lives here - never hard-coded inside the
    business logic. Changing the gate-closure time or the evaluation window is
    a configuration change, not a code change.

HOW
    pydantic-settings reads environment variables prefixed with ``WFM_``
    (or a ``.env`` file in the backend folder) and validates their types at
    start-up, so a typo such as ``WFM_DEFAULT_GATE_CLOSURE=9.50`` fails fast
    with a clear message instead of producing wrong numbers later.
"""

from datetime import date, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

# The models the API can serve, per task (see prediction_service.py for their artifacts).
RegressorName = Literal["linear_by_level", "random_forest"]
ClassifierName = Literal["logistic_by_level", "random_forest"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WFM_",
        env_file=BACKEND_DIR / ".env",
        extra="ignore",
    )

    # ---- Business scope --------------------------------------------------
    # The brief: "data relevant only for purchase decisions made in January 2026".
    # A decision made on day D-1 buys power for delivery day D, so decisions
    # 1-31 Jan cover delivery days 2 Jan - 1 Feb.
    first_decision_date: date = date(2026, 1, 1)
    last_decision_date: date = date(2026, 1, 31)

    # Gate closure of the GB day-ahead auction (Nord Pool N2EX: 09:50 UK time
    # on D-1). The latest forecast published at or before this moment is the
    # one the trading desk could actually use.
    default_gate_closure: time = time(9, 50)

    # ---- Elexon Insights (BMRS) API ---------------------------------------
    elexon_base_url: str = "https://data.elexon.co.uk/bmrs/api/v1"
    elexon_timeout_seconds: float = 60.0
    elexon_max_retries: int = 3
    elexon_chunk_days: int = 7

    # ---- Data --------------------------------------------------------------
    # "snapshot": use the CSV files in data_dir if present, otherwise download
    #             once and save them (default - fast, reproducible, offline).
    # "live":     always download fresh data from Elexon at start-up.
    data_source: Literal["snapshot", "live"] = "snapshot"
    data_dir: Path = BACKEND_DIR / "data"
    # Trained models, metrics and the ML feature table produced by backend/ml/*.py
    ml_dir: Path = BACKEND_DIR / "ml"
    # Default model for each prediction task, approved for the current application on
    # 2026-10-26 (selection history: backend/ml/README.md). The other model of each task
    # is served for comparison. Change a default only through a new model-selection decision.
    ml_default_regressor: RegressorName = "linear_by_level"
    ml_default_classifier: ClassifierName = "logistic_by_level"

    # ---- HTTP API ----------------------------------------------------------
    # Front-end origins allowed to call the API from a browser (CORS).
    # Override with a JSON list, e.g. WFM_CORS_ORIGINS='["https://my-app.vercel.app"]'
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("cors_origins")
    @classmethod
    def _normalise_origins(cls, origins: list[str]) -> list[str]:
        # A browser's Origin header is scheme://host[:port] - never a trailing "/" - and CORS
        # compares it exactly, so "https://my-app.vercel.app/" copied from the address bar would
        # silently match nothing. Only the formatting is tidied; the list stays exactly as configured.
        return [origin.strip().rstrip("/") for origin in origins]

    @property
    def first_delivery_date(self) -> date:
        return self.first_decision_date + timedelta(days=1)

    @property
    def last_delivery_date(self) -> date:
        return self.last_decision_date + timedelta(days=1)


@lru_cache
def get_settings() -> Settings:
    """Build the settings once and reuse them (environment is read only at start-up)."""
    return Settings()

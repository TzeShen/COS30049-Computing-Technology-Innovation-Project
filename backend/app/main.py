from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from phishing_detector.predict_url import MODEL_NAME, load_artifacts


class HealthResponse(BaseModel):
    status: Literal["ready"]
    model_loaded: bool
    model_name: str
    feature_count: int

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ready",
                    "model_loaded": True,
                    "model_name": "PhiUSIIL EBM",
                    "feature_count": 26,
                }
            ]
        }
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, feature_columns = load_artifacts()
    app.state.model = model
    app.state.feature_columns = feature_columns

    yield

    app.state.model = None
    app.state.feature_columns = None


app = FastAPI(
    title="Checkie API",
    version="0.1.0",
    debug=False,
    lifespan=lifespan,
)


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Check model readiness",
    description=(
        "Confirms that the saved PhiUSIIL EBM and its 26 features "
        "loaded at startup. No request body is required."
    ),
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ready",
        model_loaded=app.state.model is not None,
        model_name=MODEL_NAME,
        feature_count=len(app.state.feature_columns),
    )
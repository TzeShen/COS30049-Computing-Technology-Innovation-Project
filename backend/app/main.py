from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from phishing_detector.predict_url import MODEL_NAME, load_artifacts, score_url


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

class AnalyzeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
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
@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    result = score_url(
        payload.url,
        model=app.state.model,
        feature_cols=app.state.feature_columns,
    )
    return {**result, "reasons": result["indicators"]}
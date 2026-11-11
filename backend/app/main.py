from contextlib import asynccontextmanager
import logging
import os
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from phishing_detector.predict_url import MODEL_NAME, load_artifacts, score_url, validate_url


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

# Add the phone's frontend origin here when testing over Wi-Fi, e.g.
# CHECKIE_FRONTEND_ORIGINS=http://127.0.0.1:5173,http://192.168.1.20:5173
_default_origins = "http://127.0.0.1:5173,http://localhost:5173"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CHECKIE_FRONTEND_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request. Enter a valid HTTP or HTTPS URL of at most 2,048 characters.",
                "details": [
                    {"field": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
                    for item in exc.errors()
                ],
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail),
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    logging.error("Unexpected Checkie API error", exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Checkie could not complete the analysis because of an unexpected server error. Please try again.",
            }
        },
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
    # Validate URL syntax as client input before model inference. Any later
    # ValueError from model/inference code is treated as an unexpected 500.
    try:
        validate_url(payload.url)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_URL", "message": str(exc)}},
        )
    result = score_url(
        payload.url,
        model=app.state.model,
        feature_cols=app.state.feature_columns,
    )
    return {**result, "reasons": result["indicators"]}
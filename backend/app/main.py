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
    url: str = Field(
        min_length=1,
        max_length=2048,
        description="A URL to classify. The API analyses the URL text and does not visit the destination.",
        examples=["https://www.example.org"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"url": "https://www.example.org"},
                {"url": "http://192.0.2.1/verify-account"},
            ]
        }
    )


class Contribution(BaseModel):
    term: str
    title: str
    values: str
    interaction: bool
    contribution: float = Field(description="EBM contribution in log-odds, not percentage points.")
    direction: str


class Indicator(BaseModel):
    title: str
    flagged: bool
    detail: str


class UrlPart(BaseModel):
    text: str
    highlighted: bool
    label: str


class AnalyzeResponse(BaseModel):
    url: str
    model_name: str
    trust_score: float = Field(ge=0, le=100)
    phishing_probability: float = Field(ge=0, le=1)
    risk_percent: float = Field(ge=0, le=100)
    prediction: Literal[0, 1]
    status: Literal["Lower predicted risk", "Potential phishing"]
    status_style: Literal["safe", "dangerous"]
    threshold: float
    top_signals: list[str]
    top_contributions: list[Contribution]
    model_contributions: list[Contribution]
    base_log_odds: float
    total_log_odds: float
    raw_features: dict[str, float | int | bool]
    indicators: list[Indicator]
    url_parts: list[UrlPart]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "url": "https://www.example.org",
                    "model_name": "PhiUSIIL EBM",
                    "trust_score": 98.4,
                    "phishing_probability": 0.016,
                    "risk_percent": 1.6,
                    "prediction": 0,
                    "status": "Lower predicted risk",
                    "status_style": "safe",
                    "threshold": 0.5,
                    "top_signals": ["HTTPS scheme: lowers the phishing score"],
                    "top_contributions": [
                        {"term": "uses_https", "title": "HTTPS scheme", "values": "HTTPS scheme: Yes", "interaction": False, "contribution": -1.2, "direction": "Lowers the phishing score"}
                    ],
                    "model_contributions": [
                        {"term": "uses_https", "title": "HTTPS scheme", "values": "HTTPS scheme: Yes", "interaction": False, "contribution": -1.2, "direction": "Lowers the phishing score"}
                    ],
                    "base_log_odds": -2.0,
                    "total_log_odds": -4.12,
                    "raw_features": {"url_length": 23, "uses_https": True},
                    "indicators": [
                        {"title": "IP address host", "flagged": False, "detail": "Not detected"}
                    ],
                    "url_parts": [
                        {"text": "https://www.example.org", "highlighted": False, "label": ""}
                    ]
                }
            ]
        }
    )


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "INVALID_URL",
                        "message": "Enter a valid HTTP or HTTPS address with a hostname or IP address."
                    }
                },
                {
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "Checkie could not complete the analysis because of an unexpected server error. Please try again."
                    }
                }
            ]
        }
    )


@asynccontextmanager
async def lifespan(api: FastAPI):
    model, feature_columns = load_artifacts()
    api.state.model = model
    api.state.feature_columns = feature_columns

    yield

    api.state.model = None
    api.state.feature_columns = None


api = FastAPI(
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


@api.exception_handler(RequestValidationError)
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


@api.exception_handler(StarletteHTTPException)
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


@api.exception_handler(Exception)
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

@api.get(
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
        model_loaded=api.state.model is not None,
        model_name=MODEL_NAME,
        feature_count=len(api.state.feature_columns),
    )
@api.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse a URL for phishing risk",
    description=(
        "Classifies the submitted URL using the saved PhiUSIIL EBM. "
        "The destination website is never opened or fetched. "
        "Prediction 0 means lower predicted risk; prediction 1 means potential phishing."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "The request body or URL is invalid."},
        500: {"model": ErrorResponse, "description": "An unexpected server error occurred."},
    },
)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
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
        model=api.state.model,
        feature_cols=api.state.feature_columns,
    )
    return {**result, "reasons": result["indicators"]}

app = CORSMiddleware(
    app=api,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)
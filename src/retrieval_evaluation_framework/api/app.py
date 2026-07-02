"""FastAPI application wiring and global middleware/handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

from retrieval_evaluation_framework.api import dependencies as api_dependencies
from retrieval_evaluation_framework.api.routers import benchmarking, pipeline, retrieval, system
from retrieval_evaluation_framework.api.schemas import ErrorResponse
from retrieval_evaluation_framework.logging import get_logger

LOGGER = get_logger(component="api")
_PIPELINE_CACHE = api_dependencies._PIPELINE_CACHE

app = FastAPI(
    title="Retrieval Evaluation & Optimization Framework",
    description=(
        "REST API for corpus processing, retrieval evaluation, benchmarking, "
        "reporting, and recommendation workflows."
    ),
)

app.include_router(system.router)
app.include_router(pipeline.router)
app.include_router(retrieval.router)
app.include_router(benchmarking.router)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log request metadata and execution time for every API call."""
    start = perf_counter()
    response = await call_next(request)
    duration_ms = (perf_counter() - start) * 1000
    LOGGER.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    response.headers["X-Execution-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return friendly request validation errors."""
    LOGGER.warning(
        "request_validation_failed",
        path=request.url.path,
        errors=exc.errors(),
    )
    payload = ErrorResponse(
        detail="Invalid request payload or query parameters.",
        suggestion="Check the request schema in /docs and retry with valid fields.",
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return normalized HTTP error responses."""
    LOGGER.warning(
        "http_exception",
        path=request.url.path,
        status_code=exc.status_code,
        detail=str(exc.detail),
    )
    payload = ErrorResponse(
        detail=str(exc.detail),
        suggestion="Review endpoint parameters and resource paths.",
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Hide internal stack traces while keeping errors actionable."""
    LOGGER.error(
        "unhandled_exception",
        path=request.url.path,
        error_type=type(exc).__name__,
        message=str(exc),
    )
    payload = ErrorResponse(
        detail="Internal server error.",
        suggestion="Review server logs for details and retry the request.",
    )
    return JSONResponse(status_code=500, content=payload.model_dump())

"""System-level API routes."""

from __future__ import annotations

from fastapi import APIRouter

from retrieval_evaluation_framework.api.schemas import HealthResponse

router = APIRouter(prefix="", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service liveness status."""
    return HealthResponse(status="ok", service="retrieval-evaluation-framework")

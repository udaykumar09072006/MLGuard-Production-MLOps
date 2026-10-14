"""Health and readiness probe endpoints."""

import os

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
def liveness_check():
    """Liveness probe: verifies that the FastAPI application process is alive."""
    return {"status": "healthy", "service": "MLGuard", "environment": settings.ENVIRONMENT}


@router.get("/ready", status_code=status.HTTP_200_OK)
def readiness_check(db: Session = Depends(get_db)):
    """Readiness probe: verifies that DB and storage volumes are mounted and reachable."""
    try:
        db.execute(text("SELECT 1"))
        db_ready = True
    except Exception:
        db_ready = False

    storage_ready = (
        os.path.exists(settings.DATA_STORAGE_DIR)
        and os.path.exists(settings.MODEL_STORAGE_DIR)
        and os.path.exists(settings.REFERENCE_DATA_DIR)
    )

    if db_ready and storage_ready:
        return {
            "status": "ready",
            "database": "connected",
            "storage": "writable",
        }

    return {
        "status": "degraded",
        "database": "connected" if db_ready else "unreachable",
        "storage": "ready" if storage_ready else "missing_directories",
    }

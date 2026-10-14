"""MLGuard FastAPI main application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import datasets, experiments, health, models, monitoring, predictions, training
from app.core.config import settings
from app.core.logging_config import logger
from app.db.database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables upon startup
    logger.info("Initializing database schemas...")
    Base.metadata.create_all(bind=engine)
    logger.info("MLGuard API server initialized.")
    yield
    logger.info("MLGuard API server shutting down.")


app = FastAPI(
    title="MLGuard API",
    description="Automated Machine Learning Training, Deployment, and Monitoring Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount health & readiness directly
app.include_router(health.router)

# Mount domain routes under API prefix
app.include_router(datasets.router, prefix=settings.API_PREFIX)
app.include_router(training.router, prefix=settings.API_PREFIX)
app.include_router(experiments.router, prefix=settings.API_PREFIX)
app.include_router(models.router, prefix=settings.API_PREFIX)
app.include_router(predictions.router, prefix=settings.API_PREFIX)
app.include_router(monitoring.router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=settings.APP_PORT, reload=True)

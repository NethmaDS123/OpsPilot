"""FastAPI entrypoint: mounts ticket routes and health check."""

from fastapi import FastAPI

from app.api.tickets import router as tickets_router

app = FastAPI(
    title="OpsPilot API",
    version="0.2.0",
)

app.include_router(tickets_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }

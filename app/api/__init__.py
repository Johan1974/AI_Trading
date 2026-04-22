"""HTTP API-router(s); mount in ``app.main`` via ``include_router``."""

from app.api.router import router

__all__ = ["router"]

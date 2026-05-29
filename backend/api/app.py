"""
api/app.py — FastAPI application factory.

Creates the app, registers middleware, mounts routers, and adds a global
exception handler so all unhandled errors return a JSON body instead of
a plain-text 500.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import citations, literature, research
from api.agent.router import router as agent_router
from api.limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Assistant API",
        version="1.0.0",
        description=(
            "AI-powered research, literature review, and citation finder "
            "backed by Tavily, arXiv, Semantic Scholar, and NVIDIA LLMs."
        ),
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # TODO: restrict to known origins before going to production
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SlowAPIMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(research.router,   prefix="/research",   tags=["Topic Research"])
    app.include_router(literature.router, prefix="/literature", tags=["Literature Review"])
    app.include_router(citations.router,  prefix="/citations",  tags=["Citations"])
    app.include_router(agent_router,      prefix="/agent",      tags=["Agent"])

    # ── Health check ──────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok"}

    # ── Global exception handler ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__},
        )

    return app


app = create_app()

"""FastAPI application.

  uvicorn backend.main:app --reload --port 8000      # docs at /docs
"""
from __future__ import annotations

import common  # noqa: F401  (OpenMP bootstrap — must be first)

import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import assistant, graph, jobs, matching, resumes, system
from common.privacy import configure_logging

configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("backend")

app = FastAPI(
    title="LinkedIn Assistant API",
    description="Explainable, open-weight LLM framework for job matching. Resume text is processed locally; "
                "every recommendation exposes component scores and evidence.",
    version="1.0.0",
)
origins = [o.strip() for o in os.environ.get("FRONTEND_ORIGIN", "http://localhost:8501").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"], allow_headers=["*"])


@app.exception_handler(FileNotFoundError)
async def missing_index(_: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=503, content={"detail": f"Required data not built yet: {exc}"})


@app.exception_handler(RequestValidationError)
async def validation(_: Request, exc: RequestValidationError):
    errs = [{"loc": e.get("loc"), "msg": e.get("msg")} for e in exc.errors()]  # never echo submitted values
    return JSONResponse(status_code=422, content={"detail": errs})


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": f"Internal error ({type(exc).__name__})"})


for r in (system, resumes, jobs, matching, graph, assistant):
    app.include_router(r.router)

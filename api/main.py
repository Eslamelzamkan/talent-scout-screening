# api/main.py

from __future__ import annotations

import os
from fastapi import FastAPI, Request  # pyre-ignore[21]
from fastapi.responses import JSONResponse  # pyre-ignore[21]
from fastapi.middleware.cors import CORSMiddleware  # pyre-ignore[21]
from dotenv import load_dotenv  # pyre-ignore[21]
from slowapi import _rate_limit_exceeded_handler  # pyre-ignore[21]
from slowapi.errors import RateLimitExceeded  # pyre-ignore[21]

from api.routes import router  # pyre-ignore[21]
from api.rate_limit import limiter  # pyre-ignore[21]

load_dotenv()

app = FastAPI(
    title="TalentScout API",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.max_request_body_bytes = int(os.getenv("MAX_REQUEST_BODY_BYTES", "2000000"))

# CORS — read allowed origins from env; defaults to Streamlit dev server
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your API routes
app.include_router(router, prefix="/api")  # -> /api/v1/run


@app.middleware("http")
async def enforce_request_size(request: Request, call_next):
    if request.url.path == "/api/v1/run" and request.method.upper() == "POST":
        max_bytes = int(getattr(app.state, "max_request_body_bytes", 2_000_000))
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                if int(content_length) > max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body too large. Max {max_bytes} bytes."},
                    )
            except ValueError:
                pass

        body = await request.body()
        if len(body) > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large. Max {max_bytes} bytes."},
            )

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request._receive = receive  # pyre-ignore[16]

    return await call_next(request)


@app.get("/health", tags=["health"])
def health():
    return {"ok": True, "service": "TalentScout", "version": "1.0.0"}

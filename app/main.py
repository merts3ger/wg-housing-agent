import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routers import evaluation

app = FastAPI(title="Housing Evaluation Agent", version="0.1.0")

_DEMO_API_KEY = os.getenv("DEMO_API_KEY")


@app.middleware("http")
async def demo_key_guard(request: Request, call_next):
    if _DEMO_API_KEY and request.url.path != "/health":
        if request.headers.get("X-Demo-Key") != _DEMO_API_KEY:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(evaluation.router)

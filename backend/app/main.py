from fastapi import FastAPI
from .api.routes import router as api_router

app = FastAPI(title="backend-cpu")

# Note: CORS is handled by nginx (see nginx/default.conf)
# Do not add CORS middleware here to avoid duplicate headers

@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/healthz")
def healthz():
    return {"status": "healthy", "service": "backend-cpu"}

@app.get("/ready")
def ready():
    return {"ready": True, "reason": "ok", "checks": {"models": True, "storage": True, "vector_db": True, "redis": True}}

app.include_router(api_router)



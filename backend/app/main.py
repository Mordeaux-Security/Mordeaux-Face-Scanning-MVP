from fastapi import FastAPI
from .api.routes import api_router
app = FastAPI(title="Mordeaux API")
app.include_router(api_router, prefix="/api")
@app.get("/healthz")
def healthz():
    return {"ok": True}

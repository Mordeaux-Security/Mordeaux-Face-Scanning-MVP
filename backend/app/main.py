from fastapi import FastAPI
from .api.routes import router
app = FastAPI(title="Mordeaux API")
app.include_router(router, prefix="/api")
@app.get("/healthz")
def healthz():
    return {"ok": True}

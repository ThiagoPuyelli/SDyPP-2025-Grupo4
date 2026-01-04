from threading import Thread
from fastapi import FastAPI, Response
from scheduler import scheduler
from endpoints import results, misc, tasks
from metrics import metrics_response
from starlette.middleware.proxy_headers import ProxyHeadersMiddleware

app = FastAPI(title="Blockchain Coordinator")

app.add_middleware(
    ProxyHeadersMiddleware,
    trusted_hosts="*"
)

# Incluir routers
app.include_router(tasks.router)
app.include_router(results.router)
app.include_router(misc.router)

@app.get("/metrics")
def metrics():
    return Response(content=metrics_response(), media_type="text/plain; version=0.0.4")

@app.on_event("startup")
def iniciar_coordinador():
    thread = Thread(target=scheduler, daemon=True)
    thread.start()

from threading import Thread
from fastapi import FastAPI
from scheduler import scheduler
from endpoints import transactions, results, misc
from log_config import logger

app = FastAPI(title="Blockchain Coordinator")

# Incluir routers
app.include_router(transactions.router)
app.include_router(results.router)
app.include_router(misc.router)

@app.on_event("startup")
def iniciar_coordinador():
    thread = Thread(target=scheduler, daemon=True)  # Daemon=True para que termine al cerrar la app
    thread.start()

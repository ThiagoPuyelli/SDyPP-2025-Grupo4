from threading import Thread
from fastapi import FastAPI
from scheduler import scheduler
from endpoints import results, misc, tasks

app = FastAPI(title="Blockchain Coordinator")

# Incluir routers
app.include_router(tasks.router)
app.include_router(results.router)
app.include_router(misc.router)

@app.on_event("startup")
def iniciar_coordinador():
    thread = Thread(target=scheduler, daemon=True)
    thread.start()

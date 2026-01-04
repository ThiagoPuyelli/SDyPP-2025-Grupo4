from threading import Thread
from fastapi import FastAPI
from pool import iniciar
from endpoints import endpoints

app = FastAPI(title="Mining pool")

app.include_router(endpoints.router)

@app.on_event("startup")
def iniciar_coordinador():
    thread = Thread(target=iniciar, daemon=True)
    thread.start()

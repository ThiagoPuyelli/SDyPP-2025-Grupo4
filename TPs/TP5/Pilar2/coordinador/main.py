from fastapi import FastAPI
from scheduler import coordinator_loop
from endpoints import transactions, results, misc
import asyncio
from log_config import logger

app = FastAPI(title="Blockchain Coordinator")

# Incluir routers
app.include_router(transactions.router)
app.include_router(results.router)
app.include_router(misc.router)

@app.on_event("startup")
async def startup():
    logger.info("Iniciando ciclo principal de minería")
    asyncio.create_task(coordinator_loop())

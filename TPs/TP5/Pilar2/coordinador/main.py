from fastapi import FastAPI
from scheduler import periodic_block_generation
from endpoints import transactions, results, misc
import asyncio

app = FastAPI(title="Blockchain Coordinator")

# Incluir routers
app.include_router(transactions.router)
app.include_router(results.router)
app.include_router(misc.router)

@app.on_event("startup")
async def startup():
    print("Iniciando ciclo principal de minería")
    asyncio.create_task(periodic_block_generation())

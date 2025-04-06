from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any
import importlib

try:
    tareas = importlib.import_module("tareas")
except ModuleNotFoundError:
    tareas = None

app = FastAPI()

class TareaPayload(BaseModel):
    tarea: str
    parametros: Dict[str, Any]

@app.post("/ejecutarTarea")
def ejecutar_tarea(payload: TareaPayload):
    if tareas is None:
        raise HTTPException(status_code=500, detail="El módulo 'tareas' no fue encontrado.")

    nombre_funcion = payload.tarea
    parametros = payload.parametros

    if not hasattr(tareas, nombre_funcion):
        raise HTTPException(status_code=404, detail=f"La función '{nombre_funcion}' no existe.")

    funcion = getattr(tareas, nombre_funcion)

    try:
        resultado = funcion(**parametros)
        return {"resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar la tarea: {str(e)}")

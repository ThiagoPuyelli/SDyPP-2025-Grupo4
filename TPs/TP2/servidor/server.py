from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import docker
import requests
import uuid
import time

app = FastAPI()
docker_client = docker.from_env()
print('me lsadlasdl', flush=True)

# Clases de FastAPI para manejar JSON
class Credenciales(BaseModel):
    usuario: str
    contrasena: str

class TareaRequest(BaseModel):
    credenciales: Credenciales | None = None
    imagen: str
    tarea: str
    parametros: dict | None = None


@app.post("/getRemoteTask")
def ejecutar_tarea_remota(tarea: TareaRequest):
    print(f"==> LLEGÓ TAREA: {tarea}", flush=True)
    container_name = f"tarea_{uuid.uuid4().hex[:6]}"
    try:
        # Si hay credenciales, hacer login en Docker Hub
        if tarea.credenciales:
            docker_client.login(
                username=tarea.credenciales.usuario,
                password=tarea.credenciales.contrasena
            )

        # Ejecutar contenedor con puerto mapeado dinámicamente
        container = docker_client.containers.run(
            image=tarea.imagen,
            name=container_name,
            detach=True,
            ports={'5001/tcp': None},  # Puerto interno expuesto
        )

        # Esperar que el contenedor levante
        time.sleep(3)
        container.reload()
        host_port = container.attrs['NetworkSettings']['Ports']['5001/tcp'][0]['HostPort']

        # Enviar parámetros a /ejecutarTarea del contenedor
        url = f"http://localhost:{host_port}/ejecutarTarea"
        response = requests.post(url, json=tarea.parametros)

        # Detener y eliminar el contenedor
        container.remove(force=True)

        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

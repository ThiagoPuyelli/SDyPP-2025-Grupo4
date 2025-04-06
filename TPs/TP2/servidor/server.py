from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import docker
import requests
import uuid
import time
import json

app = FastAPI()
docker_client = docker.from_env()

# Clases de FastAPI para manejar JSON
class Credenciales(BaseModel):
    usuario: str
    contrasena: str

class TareaRequest(BaseModel):
    credenciales: str | None = None
    imagen: str
    tarea: str
    parametros: dict | None = None

def cargar_clave_privada(path="clave_privada.pem"):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def descifrar_credenciales(credenciales_cifradas: str, clave_privada) -> dict:
    datos_cifrados = base64.b64decode(credenciales_cifradas)
    datos_descifrados = clave_privada.decrypt(
        datos_cifrados,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return json.loads(datos_descifrados.decode())

@app.post("/getRemoteTask")
def ejecutar_tarea_remota(tarea: TareaRequest):
    print(f"==> LLEGÓ TAREA: {tarea}", flush=True)
    container_name = f"tarea_{uuid.uuid4().hex[:6]}"
    container = None
    try:
        # Login en Docker Hub (si hay credenciales)
        if tarea.credenciales:
            clave_privada = cargar_clave_privada()
            datos = descifrar_credenciales(tarea.credenciales, clave_privada)
            try:
                docker_client.login(    
                    username=datos["usuario"],
                    password=datos["contrasena"]
                    )
            except docker.errors.APIError as e: 
                print("Error al hacer login:", e.explanation)
                raise HTTPException(status_code=401, detail="Credenciales inválidas para Docker Hub")

        # Ejecutar contenedor
        try:
            container = docker_client.containers.run(
                image=tarea.imagen,
                name=container_name,
                detach=True,
                network="tarea-net",
                auto_remove=True
            )
        except docker.errors.ImageNotFound:
            raise HTTPException(status_code=404, detail=f"La imagen '{tarea.imagen}' no fue encontrada en Docker Hub.")
        except docker.errors.APIError as api_err:
            raise HTTPException(status_code=500, detail=f"Error de Docker: {str(api_err)}")

        # Esperar que se levante
        time.sleep(2)
        container.reload()

        # Armar request
        url = f"http://{container_name}:5000/ejecutarTarea"
        payload = {
            "tarea": tarea.tarea,
            "parametros": tarea.parametros or {}
        }

        # Reintentar conexión
        for _ in range(10):
            try:
                response = requests.post(url, json=payload, timeout=3)
                if response.ok:
                    return response.json()
            except requests.RequestException as e:
                print(f"Error: {e}", flush=True)
                time.sleep(1)

        # Si no responde
        raise HTTPException(status_code=504, detail="Timeout esperando respuesta de la tarea")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception as cleanup_error:
                print(f"Error eliminando el contenedor: {cleanup_error}", flush=True)


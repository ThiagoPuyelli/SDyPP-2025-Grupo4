from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
import base64
import json
import time
import requests

def cargar_clave_publica(path="clave_publica.pem"):
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read())
    
def cifrar_credenciales(credenciales: dict, public_key) -> str:
    json_bytes = json.dumps(credenciales).encode("utf-8")
    cifrado = public_key.encrypt(
        json_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return base64.b64encode(cifrado).decode("utf-8")

def obtener_datos():
    imagen = input("Nombre de la imagen Docker (dejar vacio para un ejemplo precargado): ")

    if not imagen:
        r = {
            "credenciales": None,
            "imagen": "matiasherrneder/tarea-cliente:latest",
            "tarea": "sumar",
            "parametros": {
                "x": "2",
                "y": "3"
            }
        }
        print("Solicitando: ", r)
        return r

    tarea = input("Nombre de la tarea a ejecutar: ")

    print("Parámetros (formato clave=valor, uno por línea, enter vacío para finalizar) :")
    parametros = {}
    while True:
        linea = input()
        if not linea:
            break
        try:
            clave, valor = linea.split("=", 1)
            parametros[clave.strip()] = valor.strip()
        except ValueError:
            print("Formato inválido. Usá clave=valor.")

    usuario = input("Usuario de Docker Hub (dejar vacío si no aplica): ")
    contrasena = input("Contraseña (dejar vacío si no aplica): ")

    credenciales_dict = {
        "usuario": usuario,
        "contrasena": contrasena
    } if usuario and contrasena else None

    public_key = cargar_clave_publica()
    credenciales_cifradas = cifrar_credenciales(credenciales_dict, public_key) if credenciales_dict else None

    r = {
        "credenciales": credenciales_cifradas,
        "imagen": imagen,
        "tarea": tarea,
        "parametros": parametros
    }
    print("Solicitando: ", r)
    return r

def enviar_peticion(data):
    url = "http://servidor:5000/getRemoteTask"
    
    for intento in range(10):
        try:
            print(f"Conectando al servidor (intento {intento+1})...", flush=True)
            response = requests.post(url, json=data)
            print("Respuesta del servidor:", flush=True)
            print(response.status_code, response.text, flush=True)
            return
        except requests.exceptions.ConnectionError:
            print("Servidor no disponible aún. Reintentando en 3 segundos...", flush=True)
            time.sleep(3)

    print("No se pudo establecer conexión con el servidor.", flush=True)

if __name__ == "__main__":
    while True:
        datos = obtener_datos()
        enviar_peticion(datos)

import time
import requests
import json

def obtener_datos():
    imagen = input("Nombre de la imagen Docker: ")
    tarea = input("Nombre de la tarea a ejecutar: ")

    print("Parámetros (formato clave=valor, uno por línea, enter vacío para finalizar):")
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

    credenciales = {
        "usuario": usuario,
        "contrasena": contrasena
    } if usuario and contrasena else None

    return {
        "imagen": imagen,
        "tarea": tarea,
        "parametros": parametros,
        "credenciales": credenciales
    }

def enviar_peticion(data):
    url = "http://servidor:5000/getRemoteTask"
    
    for intento in range(10):
        try:
            print(f"Conectando al servidor (intento {intento+1})...")
            response = requests.post(url, json=data)
            print("Respuesta del servidor:")
            print(response.status_code, response.text)
            return
        except requests.exceptions.ConnectionError:
            print("Servidor no disponible aún. Reintentando en 3 segundos...")
            time.sleep(3)

    print("No se pudo establecer conexión con el servidor.")

if __name__ == "__main__":
    datos = obtener_datos()
    enviar_peticion(datos)

import socket
import json
from datetime import datetime
import random
import sys

conexiones_activas = {}

s = None    # variable para el socket

def mostrar_menu():
    print("\n===== MENÚ =====")
    print("1. Ver nodos registrados")
    print("2. Registrarme a la siguiente ventana")
    print("3. Salir")
    return input("Selecciona una opción: ").strip()

def obtener_conexion(ip_d, puerto_d):
    global s
    if s is None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ip_d, puerto_d))

def cerrar_conexion():
    global s
    if s is not None:
        s.close()
        s = None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python nodo_c.py <IP_D> <PUERTO_D>")
        sys.exit(1)

    ip_d = sys.argv[1]
    puerto_d = int(sys.argv[2])
    mi_ip = socket.gethostbyname(socket.gethostname())
    mi_puerto = random.randint(30000, 60000) # obtengo puerto aleatorio

    # me conecto al servidor
    obtener_conexion(ip_d, puerto_d)
    print("📡 Registrando nodo en D...")

    while True:
        opcion = mostrar_menu()
        if opcion == "1": # ver nodos registrados en ventana actual
            mensaje = {
                        "tipo": "ver_registrados",
                        "hora": datetime.now().isoformat(),
                        "datos": ""
                        }
            s.sendall(json.dumps(mensaje).encode())
            data = s.recv(4096)
            respuesta = json.loads(data.decode())
            if not respuesta["datos"]:
                print("⚠️ No hay nodos registrados en la ventana actual.")
            else:
                print("Nodos registrados en la ventana actual: ")
                print(respuesta["datos"])

        elif opcion == "2": # registrarme a la siguiente ventana
            mensaje = {
                        "tipo": "registrarme",
                        "hora": datetime.now().isoformat(),
                        "datos": ""
                        }
            s.sendall(json.dumps(mensaje).encode())
            data = s.recv(4096)

            print(data.decode())

            respuesta = json.loads(data.decode())
            if respuesta["tipo"] == "registrado" :
                print(f"✔️ {respuesta['datos']}")
            else:
                print("❌ Error registrandose ❌")
                print(respuesta["datos"])
        elif opcion == "3":
            print("👋 Cerrando nodo...")
            cerrar_conexion()
            break
        else:
            print("❌ Opción inválida.")

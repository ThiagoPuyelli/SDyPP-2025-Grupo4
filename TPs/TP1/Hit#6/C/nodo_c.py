import socket
import threading
import json
import time
import random
import sys

def servidor_nodo_c(mi_ip, mi_puerto):
    def manejar_cliente(conn, addr):
        with conn:
            try:
                data = conn.recv(1024)
                mensaje = json.loads(data.decode())
                print(f"📩 Saludo recibido de {addr}: {mensaje}")
                respuesta = {
                    "tipo": "respuesta",
                    "mensaje": "Hola, recibido tu saludo!"
                }
                conn.sendall(json.dumps(respuesta).encode())
            except:
                print("⚠️ Error al manejar mensaje de cliente.")

    def servidor():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((mi_ip, mi_puerto))
            s.listen()
            print(f"🟢 Nodo C escuchando en {mi_ip}:{mi_puerto}")
            while True:
                conn, addr = s.accept()
                threading.Thread(target=manejar_cliente, args=(conn, addr), daemon=True).start()

    threading.Thread(target=servidor, daemon=True).start()

def obtener_puerto_aleatorio():
    return random.randint(30000, 60000)

def registrar_con_d(ip_d, puerto_d, mi_ip, mi_puerto):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ip_d, puerto_d))
            mensaje = {
                "ip": mi_ip,
                "puerto": mi_puerto
            }
            s.sendall(json.dumps(mensaje).encode())
            data = s.recv(4096)
            lista_nodos = json.loads(data.decode())
            print(f"📋 Nodos recibidos desde D: {lista_nodos}")
            return lista_nodos
    except Exception as e:
        print(f"❌ Error al registrar con D: {e}")
        return []

def saludar_a_nodos(lista_nodos):
    for nodo in lista_nodos:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((nodo["ip"], nodo["puerto"]))
                saludo = {
                    "tipo": "saludo",
                    "mensaje": "Hola, soy un nuevo nodo!"
                }
                s.sendall(json.dumps(saludo).encode())
                data = s.recv(1024)
                print(f"🤝 Respuesta de {nodo['ip']}:{nodo['puerto']}: {json.loads(data.decode())}")
        except Exception as e:
            print(f"⚠️ No se pudo saludar a {nodo['ip']}:{nodo['puerto']} - {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python nodo_c.py <IP_D> <PUERTO_D>")
        sys.exit(1)

    ip_d = sys.argv[1]
    puerto_d = int(sys.argv[2])
    mi_ip = socket.gethostbyname(socket.gethostname())
    mi_puerto = obtener_puerto_aleatorio()

    servidor_nodo_c(mi_ip, mi_puerto)
    time.sleep(1)  # Espera a que el servidor esté listo

    otros_nodos = registrar_con_d(ip_d, puerto_d, mi_ip, mi_puerto)
    saludar_a_nodos(otros_nodos)

    print("🟡 Nodo listo. Puedes dejar este nodo corriendo para recibir mensajes.")
    while True:
        time.sleep(10)

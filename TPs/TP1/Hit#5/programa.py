import socket
import threading
import sys
import time
import json

def servidor(mi_ip, mi_puerto):
    """Función que maneja la parte del servidor, esperando saludos en formato JSON."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((mi_ip, mi_puerto))
        server_socket.listen()
        print(f"🔵 Servidor escuchando en {mi_ip}:{mi_puerto}")

        while True:
            conn, addr = server_socket.accept()
            threading.Thread(target=manejar_cliente, args=(conn, addr), daemon=True).start()

def manejar_cliente(conn, addr):
    """Maneja la conexión con un cliente específico."""
    with conn:
        print(f"✅ Conexión establecida con {addr}")
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    print("❌ Cliente desconectado.")
                    break
                mensaje_json = json.loads(data.decode())
                print(f"📩 Mensaje recibido de {addr}: {mensaje_json['mensaje']}")

                respuesta = {
                    "tipo": "respuesta",
                    "mensaje": "Recibido!"
                }
                conn.sendall(json.dumps(respuesta).encode())

            except (socket.timeout, ConnectionResetError, json.JSONDecodeError):
                print("❌ Cliente se desconectó o envió un mensaje inválido.")
                break

def cliente(ip_destino, puerto_destino):
    """Función que maneja el cliente, enviando mensajes al otro servidor."""
    def conectar_servidor():
        while True:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((ip_destino, puerto_destino))
                print(f"🔗 Conectado a {ip_destino}:{puerto_destino}")
                return client_socket
            except (socket.error, ConnectionRefusedError):
                print("⏳ No se pudo conectar al servidor. Reintentando en 3 segundos...")
                time.sleep(3)

    client_socket = conectar_servidor()
    while True:
        mensaje_texto = input("> ")
        if mensaje_texto.lower() == "salir":
            break

        mensaje_json = {
            "tipo": "saludo",
            "mensaje": mensaje_texto
        }

        try:
            client_socket.sendall(json.dumps(mensaje_json).encode())
            data = client_socket.recv(1024)
            respuesta_json = json.loads(data.decode())
            print(f"📩 Respuesta del servidor: {respuesta_json['mensaje']}")
        except (socket.error, ConnectionResetError, BrokenPipeError, json.JSONDecodeError):
            print("⚠️ Conexión perdida o mensaje inválido. Reintentando...")
            client_socket.close()
            client_socket = conectar_servidor()

    client_socket.close()

# Validar argumentos
if len(sys.argv) < 5:
    print("Uso: python programa.py <MI_IP> <MI_PUERTO> <IP_DESTINO> <PUERTO_DESTINO>")
    sys.exit(1)

mi_ip = sys.argv[1]
mi_puerto = int(sys.argv[2])
ip_destino = sys.argv[3]
puerto_destino = int(sys.argv[4])

# Crear hilos para ejecutar servidor y cliente simultáneamente
hilo_servidor = threading.Thread(target=servidor, args=(mi_ip, mi_puerto), daemon=True)
hilo_servidor.start()

# Esperar un momento para asegurarse de que el servidor está en marcha antes de conectar el cliente
time.sleep(2)
cliente(ip_destino, puerto_destino)

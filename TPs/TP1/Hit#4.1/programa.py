import socket
import threading
import sys
import time

def servidor(mi_ip, mi_puerto):
    """Función que maneja la parte del servidor, esperando saludos."""
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
                print(f"📩 Mensaje recibido: {data.decode()}")
                conn.sendall(b"Hola desde el servidor!")
            except (socket.timeout, ConnectionResetError):
                print("❌ Cliente se desconectó abruptamente.")
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
        message = input("> ")
        if message.lower() == "salir":
            break
        try:
            client_socket.sendall(message.encode())
            data = client_socket.recv(1024)
            print(f"📩 Servidor respondió: {data.decode()}")
        except (socket.error, ConnectionResetError, BrokenPipeError):
            print("⚠️ Conexión perdida. Intentando reconectar...")
            client_socket.close()
            client_socket = conectar_servidor()
    
    client_socket.close()

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
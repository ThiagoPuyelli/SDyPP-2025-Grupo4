import socket
import sys
import time

if len(sys.argv) < 2:
    print("Uso: python client.py <IP_SERVIDOR>")
    sys.exit(1)

HOST = sys.argv[1]  # IP del servidor recibida como argumento
PORT = 12345

def conectar_servidor():
    """Intenta conectarse al servidor con reintentos en caso de fallo."""
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))
            print("Conectado al servidor.")
            return client_socket
        except (socket.error, ConnectionRefusedError):
            print("No se pudo conectar al servidor. Reintentando en 3 segundos...")
            time.sleep(3)

client_socket = conectar_servidor()

while True:
    message = input("> ")
    if message.lower() == "salir":
        break

    try:
        client_socket.sendall(message.encode())
        data = client_socket.recv(1024)
        print(f"Servidor respondió: {data.decode()}")
        break
    except (socket.error, ConnectionResetError, BrokenPipeError):
        print("Conexión perdida. Intentando reconectar...")
        client_socket.close()
        client_socket = conectar_servidor()

client_socket.close()
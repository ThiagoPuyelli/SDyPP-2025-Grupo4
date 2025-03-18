import socket
import time

HOST = 'tcp_server'  # Dirección del servidor
PORT = 65432         # Mismo puerto que el servidor

def conectar():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            # Intentar conectar al servidor
            print("Intentando conectar al servidor...", flush=True)
            client_socket.connect((HOST, PORT))
            print("Conexión exitosa al servidor", flush=True)
            
            client_socket.sendall(b"Hola, servidor!")
            print("Mensaje enviado al servidor", flush=True)
            
            data = client_socket.recv(1024).decode()
            if data:
                print(f"Servidor responde: {data}", flush=True)
            else:
                print("No se recibió respuesta del servidor.", flush=True)
        except ConnectionRefusedError:
            print("No se pudo conectar al servidor. Intentando nuevamente...", flush=True)
            return False
        return True

# Intentar conectarse
if not conectar():
    # Esperar un momento antes de reconectar
    time.sleep(2)

# Reintentar la conexión después de que el servidor cierre la conexión
if not conectar():
    print("El servidor no está disponible después de varios intentos.", flush=True)

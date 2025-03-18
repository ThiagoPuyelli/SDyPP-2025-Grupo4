import socket
import threading
import os
import time

def handle_server(HOST, PORT):
    """Función que maneja la parte del servidor."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Servidor escuchando en {HOST}:{PORT}")
        
        conn, addr = server_socket.accept()
        with conn:
            print(f"Conexión establecida desde {addr}")
            data = conn.recv(1024).decode()
            print(f"Cliente dice: {data}")
            conn.sendall(b"Hola, cliente. Conexion exitosa")

def handle_client(HOST, PORT):
    time.sleep(2)
    """Función que maneja la parte del cliente."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        client_socket.sendall(b"Hola, servidor!")
        data = client_socket.recv(1024).decode()
        print(f"Servidor responde: {data}")

def main():
    # Leer variables de entorno
    server_ip = os.getenv("SERVER_IP")
    server_port = int(os.getenv("SERVER_PORT"))
    client_ip = os.getenv("CLIENT_IP")
    client_port = int(os.getenv("CLIENT_PORT"))

    # Crear hilos para servidor y cliente
    server_thread = threading.Thread(target=handle_server, args=(server_ip, server_port))
    client_thread = threading.Thread(target=handle_client, args=(client_ip, client_port))
    
    # Iniciar los hilos
    server_thread.start()
    client_thread.start()
    
    # Esperar a que los hilos terminen
    server_thread.join()
    client_thread.join()

if __name__ == "__main__":
    main()

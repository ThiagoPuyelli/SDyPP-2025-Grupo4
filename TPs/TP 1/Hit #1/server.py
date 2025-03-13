# Servidor TCP (B)
import socket

HOST = '127.0.0.1'  # Escucha en localhost
PORT = 65432        # Puerto de escucha

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


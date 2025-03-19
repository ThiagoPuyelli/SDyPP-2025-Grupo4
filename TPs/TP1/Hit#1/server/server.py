# server.py
import socket

HOST = "0.0.0.0"  # Escucha en todas las interfaces
PORT = 12345       # Puerto de escucha

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Servidor escuchando en {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    with conn:
        print(f"Conexión establecida con {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"Cliente dijo: {data.decode()}")
            conn.sendall(b"Hola, Cliente A! Soy el Servidor B.")
            break
# Cliente TCP (A)
import socket

HOST = 'tcp_server'  # Dirección del servidor
PORT = 65432        # Mismo puerto que el servidor

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))
    client_socket.sendall(b"Hola, servidor!")
    data = client_socket.recv(1024).decode()
    print(f"Servidor responde: {data}")
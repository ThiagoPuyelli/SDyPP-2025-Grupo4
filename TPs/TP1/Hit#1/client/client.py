import socket
import sys

if len(sys.argv) < 2:
    print("Uso: python client.py <IP_SERVIDOR>")
    sys.exit(1)

HOST = sys.argv[1]  # IP del servidor recibida como argumento
PORT = 12345

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))
    print("Conectado al servidor. Escriba su mensaje:")

    while True:
        message = input("> ")
        if message.lower() == "salir":
            break
        client_socket.sendall(message.encode())
        data = client_socket.recv(1024)
        print(f"Servidor respondió: {data.decode()}")
        break
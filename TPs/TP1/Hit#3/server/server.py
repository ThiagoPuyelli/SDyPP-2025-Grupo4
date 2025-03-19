import socket

HOST = "0.0.0.0"  # Escucha en todas las interfaces
PORT = 12345       # Puerto de escucha

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Servidor escuchando en {HOST}:{PORT}")

    while True:
        print("Esperando conexión de un cliente...")
        conn, addr = server_socket.accept()
        with conn:
            print(f"Conexión establecida con {addr}")
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        print("Cliente desconectado.")
                        break
                    print(f"Cliente dijo: {data.decode()}")
                    conn.sendall(b"Hola, Cliente A! Soy el Servidor B.")
                except ConnectionResetError:
                    print("Cliente se desconectó abruptamente.")
                    break  # Volver a esperar nuevas conexiones

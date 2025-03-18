import socket

HOST = '0.0.0.0'  # Escucha en todas las interfaces
PORT = 65432       # Puerto de escucha

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Servidor escuchando en {HOST}:{PORT}", flush=True)
    
    while True:
        conn, addr = server_socket.accept()
        with conn:
            print(f"Conexión establecida desde {addr}", flush=True)
            data = conn.recv(1024).decode()
            print(f"Cliente dice: {data}", flush=True)
            conn.sendall(b"Hola, cliente. Conexion exitosa")
        
        # Cerrar la conexión después de responder al cliente
        print("Conexión cerrada con el cliente.", flush=True)

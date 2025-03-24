import socket
import threading
import json

nodos_registrados = []  # Lista en RAM con los nodos C registrados


def manejar_cliente(conn, addr):
    with conn:
        try:
            print(f"🚪 Conexión recibida de {addr}, esperando datos...")

            while True:  # Mantener el servidor esperando por datos en un bucle
                data = conn.recv(1024)  # Esperar datos del cliente

                if not data:
                    print(f"❌ No se recibieron datos de {addr}")
                    break  # Si no hay datos, terminar la conexión

                # Procesar los datos
                info_nodo = json.loads(data.decode())

                # Agregar a la lista si no existe ya
                if info_nodo not in nodos_registrados:
                    nodos_registrados.append(info_nodo)
                    print(f"🆕 Nodo registrado: {info_nodo}\n")
                else:
                    print(f"📩Recibí una consulta de la lista de nodos de: {info_nodo}\n")

                # Responder con la lista de nodos (excepto el mismo)
                otros_nodos = [n for n in nodos_registrados if n != info_nodo]

                conn.sendall(json.dumps(otros_nodos).encode())  # Responder al cliente

        except Exception as e:
            print(f"❌ Error manejando cliente {addr}: {e}")

# def manejar_cliente(conn, addr):
#     with conn:
#         try:
#             data = conn.recv(1024)
#             if not data:
#                 print(f"❌ No se recibieron datos de {addr}")
#                 return
#             info_nodo = json.loads(data.decode())      

#             # Agregar a la lista si no existe ya
#             if info_nodo not in nodos_registrados:
#                 nodos_registrados.append(info_nodo)
#                 print(f"🆕 Nodo registrado: {info_nodo}")
#             else: 
#                 print(f"Recibi un mensaje de: {info_nodo}")

#             # Responder con la lista de nodos (excepto el mismo)
#             otros_nodos = [n for n in nodos_registrados if n != info_nodo]
#             conn.sendall(json.dumps(otros_nodos).encode())

#         except Exception as e:
#             print(f"❌ Error manejando cliente {addr}: {e}")

def servidor_registro(ip, puerto):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, puerto))
        s.listen()
        print(f"📒 Servidor de registro D escuchando en {ip}:{puerto}")

        while True:
            conn, addr = s.accept()
            threading.Thread(target=manejar_cliente, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python registro_d.py <IP> <PUERTO>")
        sys.exit(1)

    ip = sys.argv[1]
    puerto = int(sys.argv[2])
    servidor_registro(ip, puerto)

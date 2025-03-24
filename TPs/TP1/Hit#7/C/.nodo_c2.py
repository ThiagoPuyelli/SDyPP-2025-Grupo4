import socket
import threading
import json
import time
import random
import sys

conexiones_activas = {}

def servidor_nodo(mi_ip, mi_puerto):
    def manejar_cliente(conn, addr):
        with conn:
            while True:
                try:
                    data = conn.recv(1024)
                    mensaje = json.loads(data.decode())
                    tipo = mensaje.get("tipo", "")

                    if tipo == "saludo":
                        print(f"\n👋 Saludo de {addr}: {mensaje['mensaje']}")
                    elif tipo == "mensaje":
                        print(f"\n💬 Mensaje de {addr}: {mensaje['mensaje']}")
                    else:
                        print(f"\n📨 Mensaje desconocido: {mensaje}")
                except:
                    print("\n⚠️ Error al manejar mensaje entrante.")

    def servidor():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((mi_ip, mi_puerto))
            s.listen()
            print(f"🟢 Nodo C escuchando en {mi_ip}:{mi_puerto}")
            while True:
                conn, addr = s.accept()
                threading.Thread(target=manejar_cliente, args=(conn, addr), daemon=True).start()

    threading.Thread(target=servidor, daemon=True).start()

def obtener_puerto_aleatorio():
    return random.randint(30000, 60000)

def registrar_con_d(ip_d, puerto_d, mi_ip, mi_puerto):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ip_d, puerto_d))
            mensaje = {
                "ip": mi_ip,
                "puerto": mi_puerto
            }
            s.sendall(json.dumps(mensaje).encode())
            data = s.recv(4096)
            lista_nodos = json.loads(data.decode())
            return lista_nodos
    except Exception as e:
        print(f"❌ Error al registrar con D: {e}")
        return []

def conectar_y_saludar(nodo):
    ip, puerto = nodo["ip"], nodo["puerto"]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ip, puerto))
        saludo = {
            "tipo": "saludo",
            "mensaje": "¡Hola! Soy un nuevo nodo en la red."
        }
        s.sendall(json.dumps(saludo).encode())
        conexiones_activas[(ip, puerto)] = s
        print(f"🤝 Conectado y saludado a {ip}:{puerto}")
    except Exception as e:
        print(f"⚠️ No se pudo conectar a {ip}:{puerto} - {e}")

def mostrar_menu():
    print("\n===== MENÚ =====")
    print("1. Ver nodos conectados")
    print("2. Enviar mensaje a un nodo")
    print("3. Enviar mensaje global (a todos)")
    print("4. Salir")
    return input("Selecciona una opción: ").strip()

def obtener_nodos_actualizados(ip_d, puerto_d, mi_ip, mi_puerto):
    nodos = registrar_con_d(ip_d, puerto_d, mi_ip, mi_puerto)
    return [n for n in nodos if not (n["ip"] == mi_ip and n["puerto"] == mi_puerto)]

def elegir_nodo(otros_nodos):
    if not otros_nodos:
        print("⚠️ No hay otros nodos conectados.")
        return None

    print("\nNodos disponibles:")
    for i, nodo in enumerate(otros_nodos):
        print(f"{i + 1}. {nodo['ip']}:{nodo['puerto']}")
    
    try:
        idx = int(input("Elegí el número del nodo destino: ")) - 1
        return otros_nodos[idx] if 0 <= idx < len(otros_nodos) else None
    except:
        print("❌ Selección inválida.")
        return None

def enviar_mensaje(nodo, mensaje):
    ip, puerto = nodo["ip"], nodo["puerto"]
    clave = (ip, puerto)
    s = conexiones_activas.get(clave)

    try:
        if s is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((ip, puerto))
            conexiones_activas[clave] = s

        paquete = {
            "tipo": "mensaje",
            "mensaje": mensaje
        }
        s.sendall(json.dumps(paquete).encode())
        print(f"✅ Mensaje enviado a {ip}:{puerto}")
    except Exception as e:
        print(f"❌ No se pudo enviar el mensaje a {ip}:{puerto} - {e}")
        if clave in conexiones_activas:
            del conexiones_activas[clave]

def enviar_mensaje_global(ip_d, puerto_d, mi_ip, mi_puerto, mensaje):
    nodos = obtener_nodos_actualizados(ip_d, puerto_d, mi_ip, mi_puerto)
    if not nodos:
        print("⚠️ No hay otros nodos conectados.")
        return

    enviados = 0
    for nodo in nodos:
        if nodo["ip"] == mi_ip and nodo["puerto"] == mi_puerto:
            continue  # No te mandes mensaje a vos mismo

        try:
            enviar_mensaje(nodo, mensaje)
            enviados += 1
        except Exception as e:
            print(f"❌ No se pudo enviar a {nodo['ip']}:{nodo['puerto']} - {e}")

    if enviados > 0:
        print(f"✅ Mensaje global enviado a {enviados} nodo(s).")
    else:
        print("⚠️ No se pudo enviar el mensaje a ningún nodo.")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python nodo_c.py <IP_D> <PUERTO_D>")
        sys.exit(1)

    ip_d = sys.argv[1]
    puerto_d = int(sys.argv[2])
    mi_ip = socket.gethostbyname(socket.gethostname())
    mi_puerto = obtener_puerto_aleatorio()

    servidor_nodo(mi_ip, mi_puerto)
    time.sleep(1)

    print("📡 Registrando nodo en D...")
    otros_nodos = obtener_nodos_actualizados(ip_d, puerto_d, mi_ip, mi_puerto)

    for nodo in otros_nodos:
        conectar_y_saludar(nodo)

    while True:
        opcion = mostrar_menu()
        if opcion == "1":
            otros_nodos = obtener_nodos_actualizados(ip_d, puerto_d, mi_ip, mi_puerto)
            if otros_nodos:
                print("\n🌐 Nodos registrados:")
                for nodo in otros_nodos:
                    print(f"- {nodo['ip']}:{nodo['puerto']}")
            else:
                print("⚠️ No hay otros nodos conectados.")
        elif opcion == "2":
            otros_nodos = obtener_nodos_actualizados(ip_d, puerto_d, mi_ip, mi_puerto)
            nodo_destino = elegir_nodo(otros_nodos)
            if nodo_destino:
                mensaje = input("Escribí tu mensaje: ")
                enviar_mensaje(nodo_destino, mensaje)
        elif opcion == "3":
            mensaje = input("Mensaje global: ")
            enviar_mensaje_global(ip_d, puerto_d, mi_ip, mi_puerto, mensaje)
        elif opcion == "4":
            print("👋 Cerrando nodo...")
            break
        else:
            print("❌ Opción inválida.")

    for sock in conexiones_activas.values():
        try:
            sock.close()
        except:
            pass

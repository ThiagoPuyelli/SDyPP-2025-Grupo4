import socket
import threading
import json
from datetime import datetime, timedelta
import time

reg_actual = []
reg_prox = []

def manejar_cliente(conn, addr):
    global reg_actual, reg_prox
    with conn:
        try:
            print(f"🚀 Cliente conectado {addr[0]}")
            while True:  # Mantener el servidor esperando por datos
                data = conn.recv(1024)

                if not data:
                    raise Exception(f"No se recibieron datos")
                paquete = json.loads(data.decode())
                

                if paquete['tipo'] == 'registrarme':
                    # si ya esta registrado
                    if addr[0] in reg_prox:
                        print(f"⚠️ El cliente {addr[0]} ya se encuentra registrado para la siguiente ventana")
                        respuesta = {
                                "tipo": "error",
                                "hora": datetime.now().isoformat(),
                                "datos": "Ya esta registrado a la proxima ventana"
                            }
                        conn.sendall(json.dumps(respuesta).encode())
                    
                    else:
                        t = ventana_objetivo(datetime.fromisoformat(paquete["hora"]))
                        
                        # si el cliente envia el mensaje para conectarse a una sesion que ya comenzo lo rechazamos
                        if t != ventana_objetivo(datetime.now()):
                            respuesta = {
                                "tipo": "error",
                                "hora": datetime.now().isoformat(),
                                "datos": "La peticion corresponde a la inscripcion de una ventana ya comenzada"
                            }
                            conn.sendall(json.dumps(respuesta).encode())
                            print(f"No registrado, la peticion corresponde a la inscripcion de una ventana ya comenzada")

                        else:
                            # registro cliente
                            reg_prox.append(addr[0])
                            
                            # timer para cerrar la conexion una vez termina la ventana 
                            hilo_cierre = threading.Thread(target=timer_conexion_cliente, args=(conn, t))
                            hilo_cierre.start()

                            # escribo en el log
                            with open("/app/logs/application.log", "a") as log_file:
                                log_file.write(f"{datetime.now()} - Se registro el cliente: {addr[0]}, a la ventana {t}\n")

                            # respondo al cliente
                            respuesta = {
                                "tipo": "registrado",
                                "hora": datetime.now().isoformat(),
                                "datos": "Registrado a la ventana: "+ t.isoformat()
                            }
                            conn.sendall(json.dumps(respuesta).encode())
                            print(f"✔️ El cliente {addr[0]} fue registrado en la ventana siguiente")
            
                elif paquete['tipo'] == 'ver_registrados':
                    respuesta = {
                        "tipo": "lista",
                        "hora": datetime.now().isoformat(),
                        "datos": reg_actual
                    }
                    conn.sendall(json.dumps(respuesta).encode())

                else: raise Exception("Mensaje invalido")
        
        except Exception as e:
            print(f"❌ Error manejando cliente {addr[0]}: {e}")
        
        conn.close()

def timer_conexion_cliente(conn, t):
    time.sleep((t + timedelta(minutes=1) - datetime.now()).total_seconds())
    conn.close()    

def ventana_objetivo(t):
    return t.replace(second=0, microsecond=0) + timedelta(minutes=1)

def cambiar_ventana():
    global reg_actual, reg_prox
    while True:
        time.sleep(60 - (time.time() % 60 )) # duerme hasta el cambio de minuto
        print(f"Nuevo minuto: {time.strftime('%H:%M:%S')}")
        reg_actual = reg_prox
        reg_prox = []

def servidor_registro(ip, puerto):
    # escribo en el log
    with open("/app/logs/application.log", "a") as log_file:
        log_file.write(f"\n---- Se levanto un nuevo servidor de registro ----\n")

    # levanto el hilo que cambia de ventana cada 60s
    threading.Thread(target=cambiar_ventana).start()

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

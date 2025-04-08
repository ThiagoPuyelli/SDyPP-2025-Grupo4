import requests
import time
import json

def medir_respuesta_http(url, intentos=5, intervalo=1):
    resultados = {
        "url": url,
        "intentos": intentos,
        "intervalo_s": intervalo,
        "resultados": [],
        "errores": 0,
        "promedio_ms": None
    }

    tiempos = []
    time.sleep(2)

    for i in range(intentos):
        intento = {"n": i + 1, "status": None, "tiempo_ms": None, "error": None}
        try:
            inicio = time.time()
            response = requests.get(url, timeout=5)
            fin = time.time()

            duracion = (fin - inicio) * 1000  # milisegundos
            intento["status"] = response.status_code
            intento["tiempo_ms"] = round(duracion, 2)
            tiempos.append(duracion)
        except requests.RequestException as e:
            intento["error"] = str(e)
            resultados["errores"] += 1
        resultados["resultados"].append(intento)

        time.sleep(intervalo)

    if tiempos:
        promedio = sum(tiempos) / len(tiempos)
        resultados["promedio_ms"] = round(promedio, 2)

    return json.dumps(resultados, indent=4)

def echo(e):
    return e
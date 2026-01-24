import subprocess
import json
from log_config import logger
import base64

def conseguirHash(hash_objetivo, cadena, minimo, maximo, stop):
    """
    Ejecuta el binario CUDA ya compilado para buscar un nonce válido.
    Devuelve (nonce, hash) o (None, None)
    """

    rango = 100_000_000
    min_actual = minimo
    encontrado = False
    resultado_nonce = None
    resultado_hash = None

    while min_actual <= maximo and not stop.is_set() and not encontrado:
        max_actual = min(min_actual + rango - 1, maximo)

        # logger.info(
        #     f"GPU search | min={min_actual} max={max_actual} "
        #     f"hash={hash_objetivo} cadena={cadena}"
        # )

        cadena_b64 = base64.b64encode(cadena.encode()).decode()

        comando = [
            "./mineroGPU/fuerzaBruta",
            hash_objetivo,
            cadena_b64,
            str(min_actual),
            str(max_actual),
        ]
        # logger.info(f"Ejecutando comando: {' '.join(comando)}")
        proceso = subprocess.run(
            comando,
            capture_output=True,
            text=True,
        )

        if proceso.returncode != 0:
            logger.error("Error ejecutando fuerzaBruta (CUDA)")
            logger.error(proceso.stderr)
            return None, None

        try:
            salida = json.loads(proceso.stdout)
            logger.debug(f"Salida CUDA: {salida}")
        except json.JSONDecodeError:
            logger.error("Salida CUDA inválida:")
            logger.error(proceso.stdout)
            return None, None

        if salida.get("found"):
            encontrado = True
            resultado_nonce = salida.get("nonce")
            resultado_hash = salida.get("hash")
        else:
            min_actual = max_actual + 1

    if encontrado:
        return resultado_nonce, resultado_hash

    return None, None
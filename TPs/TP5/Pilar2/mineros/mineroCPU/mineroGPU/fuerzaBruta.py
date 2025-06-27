import subprocess
import json
from log_config import logger


def conseguirHash(hash, cadena, min, max, stop):
    file_path= 'json_output.txt'
    with open(file_path, 'w') as archivo:
        json.dump({"numero": 0, "hash_md5_result": ""}, archivo)

    # Comando para compilar el archivo CUDA
    compile_command = ['nvcc', 'mineroGPU/fuerzaBruta.cu', '-o', 'mineroGPU/f']

    # Ejecutar el comando de compilación
    compile_process = subprocess.run(compile_command, capture_output=True, text=True)
    # Verificar si la compilación fue exitosa
    if compile_process.returncode != 0:
        logger.info("Error al compilar el archivo CUDA:")
        logger.info(compile_process.stderr)
        return None, None
    
    rango = 1000000
    minActual = min
    maxActual = minActual + rango
    encontrado = False
    resultado = None
    
    while minActual < max and not stop.is_set() and not encontrado:
        logger.info(f"MIN ACTUAL, MAX, HASH, CADENA: {minActual} {max} {hash} {cadena}")
        execute_command = ['./mineroGPU/f', hash, cadena, str(minActual), str(maxActual)]
        
        execute_process = subprocess.run(execute_command, capture_output=True, text=True)
        
        with open(file_path, 'r') as archivo:
            contenido = archivo.read()
        
        
        resultado = json.loads(contenido)
        
        if not(resultado['hash_md5_result'] == ""):
            encontrado = True
        else:
            minActual = maxActual + 1
            siguienteMax = minActual + rango
            maxActual = max if siguienteMax > max else siguienteMax
    
    if encontrado:
        return resultado['numero'], resultado['hash_md5_result']
    else:
        return None, None 
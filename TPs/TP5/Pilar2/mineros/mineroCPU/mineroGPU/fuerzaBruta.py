import subprocess
import json
from log_config import logger


def conseguirHash(hash, cadena, min, max, stop):
    logger.info("ACA LLEGO 1")
    file_path= 'json_output.txt'
    with open(file_path, 'w') as archivo:
        json.dump({"numero": 0, "hash_md5_result": ""}, archivo)

    logger.info("ACA LLEGO 2")
    # Comando para compilar el archivo CUDA
    compile_command = ['nvcc', 'mineroGPU/fuerzaBruta.cu', '-o', 'mineroGPU/f']

    logger.info("ACA LLEGO 3")
    # Ejecutar el comando de compilación
    compile_process = subprocess.run(compile_command, capture_output=True, text=True)
    # Verificar si la compilación fue exitosa
    logger.info("ACA LLEGO 4")
    if compile_process.returncode != 0:
        logger.info("Error al compilar el archivo CUDA:")
        logger.info(compile_process.stderr)
        return None, None
    logger.info("ACA LLEGO 5")
    
    rango = 10000
    minActual = min
    maxActual = minActual + rango
    encontrado = False
    resultado = None
    
    logger.info("ACA LLEGO 6")
    while minActual < max and not stop.is_set() and not encontrado:
        logger.info("INCIA WHILE 1")
        logger.info(f"MIN ACTUAL, MAX, HASH, CADENA: {minActual} {max} {hash} {cadena}")
        execute_command = ['./mineroGPU/f', hash, cadena, str(minActual), str(maxActual)]
        
        logger.info("INCIA WHILE 2")
        execute_process = subprocess.run(execute_command, capture_output=True, text=True)
        
        logger.info("INCIA WHILE 3")
        with open(file_path, 'r') as archivo:
            contenido = archivo.read()
        
        logger.info(f"{contenido} CONTENIDO")
        
        resultado = json.loads(contenido)
        
        if not(resultado['hash_md5_result'] == ""):
            logger.info(f"ACA?, {resultado}")
            encontrado = True
        else:
            logger.info(f"O ACA? {resultado}")
            minActual = maxActual + 1
            siguienteMax = minActual + rango
            maxActual = max if siguienteMax > max else siguienteMax
    
    logger.info(f"{resultado} RESULTADO")
    if encontrado:
        return resultado['numero'], resultado['hash_md5_result']
    else:
        return None, None 
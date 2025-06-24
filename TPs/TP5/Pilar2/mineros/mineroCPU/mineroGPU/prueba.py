import subprocess
import json

def conseguirHashPrueba(hash, cadena, min, max):
    file_path= 'json_output.txt'
    with open(file_path, 'w') as archivo:
        json.dump({"numero": 0, "hash_md5_result": ""}, archivo)

    # Comando para compilar el archivo CUDA
    compile_command = ['nvcc', 'md5.cu', '-o', 'md5']

    # Ejecutar el comando de compilación
    compile_process = subprocess.run(compile_command, capture_output=True, text=True)
    # Verificar si la compilación fue exitosa
    if compile_process.returncode != 0:
        print("Error al compilar el archivo CUDA:")
        print(compile_process.stderr)
        return
    
    rango = 10000
    minActual = min
    maxActual = minActual + rango
    encontrado = False
    resultado = None
    
    while minActual < max and not encontrado:
        execute_command = ['./md5', hash, cadena, str(minActual), str(maxActual)]
        
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
    
        return resultado['numero'], resultado['hash_md5_result'] if encontrado else None, None
    
    return resultado['numero'], resultado['hash_md5_result'] if encontrado else None, None 

from fuerzaBruta import conseguirHash

conseguirHashPrueba("fafafa", "HOLA", 0, 1000000)
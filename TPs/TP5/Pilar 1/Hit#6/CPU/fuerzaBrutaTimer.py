import hashlib
import time

def calcular_md5(texto):
    # Crear un objeto hash MD5
    hash_md5 = hashlib.md5()
    
    # Actualizar el hash con el texto codificado en bytes (UTF-8 por defecto)
    hash_md5.update(texto.encode('utf-8'))
    
    # Obtener el hash en formato hexadecimal
    return hash_md5.hexdigest()

def conseguirHash(hash, cadena, max):
    for x in range(0, max):
        texto = cadena + str(x)
        resultado = calcular_md5(texto)
        if resultado.startswith(hash): 
            return x, resultado
    return None, None

# Entrada de usuario
hash = input("Ingresa el hash: ")
cadena = input("Ingresa la cadena: ")
maximo = int(input("Ingresa el maximo: "))

# Inicio del cronómetro
inicio = time.time()

# Ejecutar la búsqueda
numero_encontrado, hash_resultado = conseguirHash(hash, cadena, maximo)

# Fin del cronómetro
fin = time.time()
tiempo_ejecucion = fin - inicio

# Mostrar resultados
if hash_resultado:
    print(f"\nNúmero encontrado: {numero_encontrado}")
    print(f"Hash MD5: {hash_resultado}")
else:
    print("\nNo se encontró el resultado")

# Mostrar métricas de tiempo
print(f"\nTiempo de ejecución: {tiempo_ejecucion:.6f} segundos")
if tiempo_ejecucion > 0:
    hashes_por_segundo = maximo / tiempo_ejecucion
    print(f"Velocidad: {hashes_por_segundo:,.2f} hashes/segundo")
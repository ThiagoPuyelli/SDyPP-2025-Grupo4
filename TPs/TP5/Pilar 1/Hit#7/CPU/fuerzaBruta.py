import hashlib

def calcular_md5(texto):
    # Crear un objeto hash MD5
    hash_md5 = hashlib.md5()
    
    # Actualizar el hash con el texto codificado en bytes (UTF-8 por defecto)
    hash_md5.update(texto.encode('utf-8'))
    
    # Obtener el hash en formato hexadecimal
    return hash_md5.hexdigest()

def conseguirHash (hash, cadena, min, max):
    for x in range(min, max):
        texto = cadena + str(x)
        resultado = calcular_md5(texto)
        if (resultado.startswith(hash)): 
            return resultado
        

hash = input("Ingresa el hash: ")
cadena = input("Ingresa la cadena: ")
min = int(input("Ingresa el minimo: "))
max = int(input("Ingresa el maximo: "))
hash_resultado = conseguirHash(hash, cadena, min, max)
if hash_resultado:
    print(f"Resultado: {hash_resultado}")
else:
    print("No se encontro el resultado")
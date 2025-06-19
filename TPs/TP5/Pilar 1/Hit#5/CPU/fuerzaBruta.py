import hashlib

def calcular_md5(texto):
    # Crear un objeto hash MD5
    hash_md5 = hashlib.md5()
    
    # Actualizar el hash con el texto codificado en bytes (UTF-8 por defecto)
    hash_md5.update(texto.encode('utf-8'))
    
    # Obtener el hash en formato hexadecimal
    return hash_md5.hexdigest()

def conseguirHash (hash, cadena, max):
    for x in range(0, max):
        texto = cadena + str(x)
        resultado = calcular_md5(texto)
        if (resultado.startswith(hash)): 
            return [resultado, x]
        

hash = input("Ingresa el hash: ")
cadena = input("Ingresa la cadena: ")
max = int(input("Ingresa el maximo: "))
[hash_resultado, nonce] = conseguirHash(hash, cadena, max)
if hash_resultado:
    print(f"Resultado: {hash_resultado}")
    print(f"Nonce: {nonce}")
else:
    print("No se encontro el resultado")
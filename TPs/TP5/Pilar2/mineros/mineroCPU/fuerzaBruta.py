import hashlib

def calcular_md5(texto):
    # Crear un objeto hash MD5
    hash_md5 = hashlib.md5()
    
    # Actualizar el hash con el texto codificado en bytes (UTF-8 por defecto)
    hash_md5.update(texto.encode('utf-8'))
    
    # Obtener el hash en formato hexadecimal
    return hash_md5.hexdigest()

def conseguirHash(hash, cadena, min, max, stop):
    for x in range(min, max):
        if stop.is_set(): break
        
        texto = cadena + str(x)
        resultado = calcular_md5(texto)
        if resultado.startswith(hash): 
            return x, resultado
    return None, None
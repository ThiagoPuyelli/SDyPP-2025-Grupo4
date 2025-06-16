import hashlib

def calcular_md5(texto):
    # Crear un objeto hash MD5
    hash_md5 = hashlib.md5()
    
    # Actualizar el hash con el texto codificado en bytes (UTF-8 por defecto)
    hash_md5.update(texto.encode('utf-8'))
    
    # Obtener el hash en formato hexadecimal
    return hash_md5.hexdigest()

texto = input("Ingresa el texto a hashear: ")
hash_resultado = calcular_md5(texto)
print(f"MD5 de '{texto}': {hash_resultado}")
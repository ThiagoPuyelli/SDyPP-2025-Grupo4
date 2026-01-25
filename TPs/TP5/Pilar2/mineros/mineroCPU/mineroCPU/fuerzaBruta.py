from utils import calcular_md5

def conseguirHash(hash, cadena, min, max, stop):
    for x in range(min, max):
        if stop.is_set(): break
        
        texto = cadena + str(x)
        resultado = calcular_md5(texto)
        if resultado.startswith(hash): 
            return x, resultado
    return None, None
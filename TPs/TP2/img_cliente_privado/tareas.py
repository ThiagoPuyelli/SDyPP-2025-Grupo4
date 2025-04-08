def hacer_ping(url, cantidad):
    comando = ["ping", "-c", str(cantidad), url]

    try:
        resultado = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if resultado.returncode != 0:
            return {"status": "error", "mensaje": resultado.stderr}

        return {"status": "ok", "salida": resultado.stdout, **parsear_salida(resultado.stdout)}

    except Exception as e:
        return {"status": "error", "mensaje": str(e)}

def hacer_echo(string):
    return string
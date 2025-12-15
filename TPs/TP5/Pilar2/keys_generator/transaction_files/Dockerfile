# Imagen base
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
RUN pip install --no-cache-dir cryptography

# Copiar script
COPY generate_tx.py .

# Copiar keys dentro de la imagen
COPY keys /keys

# Crear directorio de salida interno (opcional: se puede montar como volumen)
RUN mkdir /output

# Comando por defecto
CMD ["python", "generate_tx.py"]

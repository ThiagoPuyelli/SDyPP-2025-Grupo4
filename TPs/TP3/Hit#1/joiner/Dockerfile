# Imagen base
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN mkdir /app/img
# Copia archivos
COPY . .

# Instala dependencias
RUN pip install --no-cache-dir fastapi uvicorn pika opencv-python numpy google-cloud-storage redis

# Expone el puerto donde correrá FastAPI
EXPOSE 8002

# Comando para ejecutar FastAPI
CMD ["python", "joiner.py"]
# Imagen base
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copia archivos
COPY . .

# Instala dependencias
RUN pip install --no-cache-dir pika opencv-python

# Comando para ejecutar el consumidor
CMD ["python", "sobel-consumer.py"]
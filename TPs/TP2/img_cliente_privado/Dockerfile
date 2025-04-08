FROM matiasherrneder/tarea-generica:latest

RUN apt-get update && apt-get install -y iputils-ping && apt-get clean

WORKDIR /app

COPY tareas.py .
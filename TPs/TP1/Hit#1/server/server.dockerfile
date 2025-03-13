# Dockerfile para el servidor
FROM python:3.9
WORKDIR /app
COPY server/server.py .
CMD ["python", "server.py"]
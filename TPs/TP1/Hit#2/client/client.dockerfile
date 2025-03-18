# Dockerfile para el cliente
FROM python:3.9
WORKDIR /app
COPY client/client.py .
CMD ["python", "client.py"]
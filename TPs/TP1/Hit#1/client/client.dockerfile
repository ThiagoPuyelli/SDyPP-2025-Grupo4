# Dockerfile.client
FROM python:3.9
WORKDIR /app
COPY client.py .
CMD ["python", "client.py"]

FROM python:3.11-slim

# Instala curl y unzip para instalar ngrok
RUN apt-get update && apt-get install -y curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Instala ngrok
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list && \
    apt-get update && apt-get install -y ngrok

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Creamos usuario no-root con UID fijo
RUN useradd -u 1001 -m appuser

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000
USER 1001
CMD ["/entrypoint.sh"]

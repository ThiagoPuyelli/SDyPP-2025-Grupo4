#!/bin/bash

echo "🔧 Iniciando FastAPI..."
uvicorn server:app --host 0.0.0.0 --port 5000 &

sleep 3

# Agrega token si lo hay
if [ -n "$NGROK_AUTHTOKEN" ]; then
  echo "🔐 Autenticando ngrok..."
  ngrok config add-authtoken "$NGROK_AUTHTOKEN"
fi

echo "🌍 Iniciando túnel ngrok..."
ngrok http 5000
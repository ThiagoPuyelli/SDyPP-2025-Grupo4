# Pasos para levantar un minero

## Setear variables de entorno
Se necesitan las siguientes variables de entorno:
```bash
# Url del host, puede conectarse a
# - Coordinador:
HOST_URI=https://blockchain.34.23.224.114.nip.io/
# - Pool de mineria:
HOST_URI=https://pool.34.23.224.114.nip.io/

# Clave publica del minero (o cualquier string para probar)
MINER_ID=mineropubkey

# Url de la cola Rabbitmq (solo si se conecta al pool)
RABBIT_HOST=35.229.51.211.nip.io
```

## Levantar mineros CPU
Usar la ultima imagen disponible en Docker Hub
```
docker run --rm --env-file .env --name minercpu matiasherrneder/blockchain-minero:052d4dc
```

## Levantar minero GPU
La version de gpu esta taggeada y subida manualmente :latest
```
docker run --rm --gpus all --env-file .env --name minergpu matiasherrneder/blockchain-minero-gpu:latest
```
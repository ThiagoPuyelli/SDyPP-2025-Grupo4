from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np
import urllib.request
import uuid
import os
import pika
import json
import time
from fastapi.responses import StreamingResponse
from google.cloud import storage

app = FastAPI()
credentials = pika.PlainCredentials('user', 'password')

client = storage.Client()
bucket = client.bucket("prueba-3fc1f.appspot.com")

for attempt in range(10):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='rabbitmq',
            port=5672,
            credentials=credentials
        ))
        print("Conectado a RabbitMQ")
        break
    except pika.exceptions.AMQPConnectionError as e:
        print(f"Intento {attempt + 1}: RabbitMQ no disponible, esperando 5 segundos...")
        time.sleep(5)
else:
    print("No se pudo conectar a RabbitMQ después de varios intentos. Abortando.")
    exit(1)
channel = connection.channel()

channel.queue_declare(queue='sobel')

def download_image(url: str, save_path: str):
    urllib.request.urlretrieve(url, save_path)

def apply_sobel_partitioned(image_path, n_parts, id):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height = img.shape[0]
    patch_height = height // n_parts

    for i in range(n_parts):
        filename = 'p' + str(i) + '.jpg'
        start_row = i * patch_height
        end_row = (i + 1) * patch_height if i != n_parts - 1 else height
        patch = img[start_row:end_row, :]
        
        upload_image_gcp(patch, id + '/' + filename)
        mensaje = json.dumps({
            'task_id': id,
            'n_image': i,
            'n_parts': n_parts
        })
        channel.basic_publish(exchange='',
                          routing_key='sobel',
                          body=mensaje)
        print(f"[Productor] Enviado: {mensaje}")

@app.get("/sobel")
def sobel_endpoint(image_url: str = Query(...), n_parts: int = Query(4)):
    id = str(uuid.uuid4())
    while (file_exist_gcp(id)):
        id = str(uuid.uuid4())

    try:
        os.makedirs(id)
        input_path = f"./{id}/input.jpg"
        download_image(image_url, input_path)
        create_dir_gcp(id)
        apply = apply_sobel_partitioned(input_path, n_parts, id)

        return {
            "message": apply if "Filtro en proceso" else "Fallo el filtro"
        }

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        return {"error": str(e)}
    
def file_exist_gcp(carpeta: str) -> bool:
    blobs = list(bucket.list_blobs(prefix=carpeta + "/"))
    return len(blobs) > 0

def create_dir_gcp(carpeta):
    blob = bucket.blob(carpeta + "/")  # Finalizar con /
    blob.upload_from_string('')

def upload_image_gcp(img, destination_blob_name):
    success, encoded_image = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("No se pudo codificar la imagen")

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpg')
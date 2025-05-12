import os
import cv2
import pika
import json
import time
import numpy as np
import requests
from google.cloud import storage

client = storage.Client()
bucket = client.bucket("bucket_sobel")

def read_img_gcp(blob_path: str):
    blob = bucket.blob(blob_path)

    image_bytes = blob.download_as_bytes()

    image_array = np.frombuffer(image_bytes, np.uint8)
    patch = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    return patch

def upload_image_gcp(img, destination_blob_name):
    success, encoded_image = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("No se pudo codificar la imagen")

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpg')

def apply_sobel_to_patch(ch, method, properties, body):#, patch, output_path="output", filename="sobel_result.jpg"):
    #try:
    datos = json.loads(body.decode())
    task_id = datos.get('task_id')
    n_image = datos.get('n_image')
    n_parts = datos.get('n_parts')
    patch = read_img_gcp(task_id + '/p' + str(n_image) + '.jpg')

    # Aplicar filtro Sobel
    sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_result = cv2.convertScaleAbs(sobel_combined)

    # Guardar la imagen
    upload_image_gcp(sobel_result, task_id + '/r' + str(n_image) + '.jpg')

    base_url = "http://joiner:8002/join"

    params = {
        "task_id": task_id,
        "n_parts": n_parts,
        "n_image": n_image
    }
    
    response = requests.get(base_url, params=params)
        
    while (response.status_code != 200):
        time.sleep(5)
        response = requests.get(base_url, params=params)
    #ch.basic_ack(delivery_tag=method.delivery_tag)
    #except Exception as e:
    #    print(f"[ERROR] Falló el procesamiento del patch: {e}")
    #    # O podrías re-publicarlo en una cola de errores si querés
    #    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

credentials = pika.PlainCredentials('user', 'password')
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

# Asegurarse de que la cola existe
channel.queue_declare(queue='sobel')

# Escuchar los mensajes
channel.basic_consume(queue='sobel',
                      on_message_callback=apply_sobel_to_patch,
                      auto_ack=True)
                      
channel.queue_declare(queue='reply-sobel')

print('[Consumidor] Esperando mensajes. Ctrl+C para salir.')
channel.start_consuming()


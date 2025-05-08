from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np
import urllib.request
import uuid
import os
import shutil
import pika
import json
import time
import uuid
from fastapi.responses import StreamingResponse

app = FastAPI()
credentials = pika.PlainCredentials('guest', 'guest')
for attempt in range(10):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost',
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

def apply_sobel_partitioned(image_path, output_path, n_parts):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height = img.shape[0]
    patch_height = height // n_parts
    o_path, part_path  = generarCarpeta()
    for i in range(n_parts):
        filename = str(i) + '.jpg'
        start_row = i * patch_height
        end_row = (i + 1) * patch_height if i != n_parts - 1 else height
        patch = img[start_row:end_row, :]
        
        p_path = os.path.join(part_path, filename)
        cv2.imwrite(p_path, patch)
        mensaje = json.dumps({
            'output_path': o_path,
            'path': p_path,
            'filename': filename
        })
        channel.basic_publish(exchange='',
                          routing_key='sobel',
                          body=mensaje)
        print(f"[Productor] Enviado: {mensaje}")

    timeout = 30  # segundos máximos de espera
    start_time = time.time()
    
    while True:
        processed_files = os.listdir(o_path)
        if len(processed_files) >= n_parts:
            break
        if time.time() - start_time > timeout:
            raise TimeoutError("Tiempo de espera excedido esperando que los consumidores terminen.")
        time.sleep(0.5)

    patches = []

    for i in range(n_parts):
        sobel_patch = cv2.imread(o_path + '/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
        patches.append(sobel_patch)
        #os.remove('tmp/' + str(i) + '.jpg')
        #os.remove('parts/' + str(i) + '.jpg')
    shutil.rmtree(o_path)
    shutil.rmtree(part_path)

    result_img = np.vstack(patches)
    #for filename in os.listdir('tmp'):
    #    file_path = os.path.join('tmp', filename)
    #    if os.path.isfile(file_path):
    #        os.remove(file_path)

    cv2.imwrite(output_path, result_img)

def generarCarpeta ():
    id = f"{uuid.uuid4().hex[:8]}"
    while os.path.exists(f'tmp{id}') or os.path.exists(f'parts{id}'):
        id = f"{uuid.uuid4().hex[:8]}"
    o_path = os.path.join(os.getcwd(), f'tmp{id}')
    part_path = os.path.join(os.getcwd(), f'parts{id}')
    os.makedirs(o_path)
    os.makedirs(part_path)
    return o_path, part_path


@app.get("/sobel")
def sobel_endpoint(image_url: str = Query(...), n_parts: int = Query(4)):
    input_path = f"./img/input_{uuid.uuid4()}.jpg"
    output_path = f"./img/output_{uuid.uuid4()}.jpg"

    try:
        download_image(image_url, input_path)
        apply_sobel_partitioned(input_path, output_path, n_parts)

        def file_iterator(path):
            with open(path, mode="rb") as file_like:
                yield from file_like
            os.remove(path)  # se elimina después de enviar
            os.remove(input_path)

        return StreamingResponse(file_iterator(output_path), media_type="image/jpeg", headers={
            "Content-Disposition": "attachment; filename=sobel_output.jpg"
        })

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        return {"error": str(e)}
#@app.get("/sobel")
#def sobel_endpoint(image_url: str = Query(...), n_parts: int = Query(4)):
#    input_path = f"./img/input_{uuid.uuid4()}.jpg"
#    output_path = f"./img/output_{uuid.uuid4()}.jpg"
#
#    try:
#        download_image(image_url, input_path)
#        apply_sobel_partitioned(input_path, output_path, n_parts)
#        imgResponse = FileResponse(output_path, media_type="image/jpeg", filename="sobel_output.jpg")
#        os.remove(input_path)
#        os.remove(output_path)
#        return imgResponse
#    except Exception as e:
#        return {"error": str(e)}
#    finally:
#        if os.path.exists(input_path):
#            os.remove(input_path)

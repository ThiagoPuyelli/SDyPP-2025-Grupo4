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
import redis
from google.cloud import storage

client = storage.Client()
bucket = client.bucket("prueba-3fc1f.appspot.com")

app = FastAPI()

def upload_image_gcp(img, destination_blob_name):
    success, encoded_image = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("No se pudo codificar la imagen")

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpg')

@app.get("/join")
def join(task_id: str, n_parts: int, n_image: int):
    keys = r.keys(task_id)
    if len(keys) < n_parts - 1:
        r.set(task_id, f'{n_image}-{n_parts}')
        return {"result": True}
    
    prefix = task_id + '/r' + str(n_image) +'.jpg'
    
    blobs = bucket.list_blobs(prefix=prefix)

    blobs.sort(key=lambda b: b.name)
    
    parts = []
    
    for blob in blobs:
        image_bytes = blob.download_as_bytes()
        
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        parts.append(image)
    
    result_img = np.vstack(parts)
    #upload_image_gcp(result_img, )

if __name__ == '__main__':
    r = redis.Redis(host='redis', port=6379, decode_responses=False)
    app.run(host='0.0.0.0', port=5002)
import os
import cv2
import pika
import json
import time

def apply_sobel_to_patch(ch, method, properties, body):#, patch, output_path="output", filename="sobel_result.jpg"):
    #try:
    datos = json.loads(body.decode())
    output_path = datos.get('output_path')
    path = datos.get('path')
    filename = datos.get('filename')
    id = datos.get('task_id')
    patch = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    os.makedirs(output_path, exist_ok=True)

    # Aplicar filtro Sobel
    sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_result = cv2.convertScaleAbs(sobel_combined)

    # Construir la ruta completa del archivo de salida
    full_output_path = os.path.join(output_path, filename)

    # Guardar la imagen
    cv2.imwrite(full_output_path, sobel_result)

    message = json.dumps({
            "task_id": id,
            "result": True
        })

    ch.basic_publish(
        exchange='',
        routing_key='reply-sobel',
        body=message
    )
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
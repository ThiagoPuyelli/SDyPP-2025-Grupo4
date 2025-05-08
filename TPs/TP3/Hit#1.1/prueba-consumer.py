# consumer.py
import pika

# Función que se ejecuta cuando llega un mensaje
def callback(ch, method, properties, body):
    numero = int(body.decode())
    resultado = numero * 5
    print(f"[Consumidor] Recibido: {numero} -> Multiplicado: {resultado}")

# Conectar a RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Asegurarse de que la cola existe
channel.queue_declare(queue='numeros')

# Escuchar los mensajes
channel.basic_consume(queue='numeros',
                      on_message_callback=callback,
                      auto_ack=True)

print('[Consumidor] Esperando mensajes. Ctrl+C para salir.')
channel.start_consuming()

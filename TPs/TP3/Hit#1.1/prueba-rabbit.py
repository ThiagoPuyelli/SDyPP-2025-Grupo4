import pika

# Conectar a RabbitMQ en localhost
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Crear la cola llamada "numeros"
channel.queue_declare(queue='numeros')

i = int(input("Ingrese un numero: "))
# Enviar varios números
while (i != 0):
    mensaje = str(i)
    channel.basic_publish(exchange='',
                          routing_key='numeros',
                          body=mensaje)
    print(f"[Productor] Enviado: {mensaje}")
    i = int(input("Ingrese un numero: "))

connection.close()
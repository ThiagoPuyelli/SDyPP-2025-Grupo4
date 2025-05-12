Bienvenido al Hit#1

# Consigna:

El operador de Sobel es una máscara que, aplicada a una imagen, permite detectar (resaltar) bordes. Este operador es una operación matemática que, aplicada a cada pixel y teniendo en cuenta los píxeles que lo rodean, obtiene un nuevo valor (color) para ese pixel. Aplicando la operación a cada píxel, se obtiene una nueva imagen que resalta los bordes.
Objetivo: 
Input: una imagen. 
proceso (Sobel).
output: una imagen filtrada.
Parte1 
Desarrollar un proceso centralizado que tome una imagen, aplique la máscara, y genere un nuevo archivo con el resultado. 

Parte2 
Desarrolle este proceso de manera distribuida donde se debe partir la imagen en n pedazos, y asignar la tarea de aplicar la máscara a N procesos distribuidos. Después deberá unificar los resultados. 

A partir de ambas implementaciones, comente los resultados de performance dependiendo de la cantidad de nodos y tamaño de imagen.

Parte3 
Mejore la aplicación del punto anterior para que, en caso de que un proceso distribuido (al que se le asignó parte de la imagen a procesar - WORKER) se caiga y no responda, el proceso principal detecte esta situación y pida este cálculo a otro proceso.

- Bien, todo eso lo corrieron “distribuido” pero “centralizado” en su propia computadora. Sin embargo, la necesidad de procesamiento para aplicar un filtro sobre imágenes (que pueden ser muy grandes) puede generar que nos encontramos con un tope de capacidad de cómputo (más allá que lo hayamos construido de forma redundante), ¿Qué opciones tenemos?
Si es local, agregar más nodos de trabajo. Sin embargo, como discutimos, eso requiere tiempo (compra, ambientación, espacio e instalación de paquetes) y produce un Costo Fijo mínimo (más allá del variable por intensidad de uso). 

- Hacer un offloading a la nube. La cual trabajaremos en este apartado. 

# Resolucion

Para solucionar esta problematica, pensamos en una infraestructura donde se ve reflejada en un diagrama en la imagen que esta en esta misma carpeta.

# Funcion de cada parte

- Rabbitmq: Gestionar la cola de mensajes entre el cliente y los consumidores.

- Reddis: Gestionar que partes de que imágenes ya fueron procesadas.

- Cliente: el cliente lo que hace es recibir las peticiones del usuario donde en base a eso tiene 2 endpoints:

/sobel: Aca se pasa una url de la imagen, el cual la imagen tiene que estar subida en la internet y la cantidad de partes a particionar y las sube a internet, una vez que el usuario usa el endpoint, el cliente particiona la imagen y agrega las tareas en la cola de rabbitmq

/get_image: En este endpoint se obtiene la imagen con un id, donde si la imagen todavia no fue terminada de procesar dara aviso de ello y sino devolvera la imagen ya procesada por el filtro de sobel

Para hacer uso de estos endpoints dejo un par de ejemplos de prueba:

- http://localhost:8000/sobel?image_url=https://imgs.search.brave.com/VyMDMutzQJTOYQZjnv9yhXb1NfyicOiLlexCkDe58KU/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zMS5z/aWduaWZpY2Fkb3Mu/Y29tL2ZvdG8vaW1h/Z2VuLWluaWNpby1j/a2UuanBnP2NsYXNz/PWFydGljbGU

- http://localhost:8000/get_image/{image-id}

En sí no dejo un ejemplo de image-id, ya que la idea es que la primer consulta devuelva el id de la imagen, una vez devuelto con eso se ejecutaria la siguiente petición.

- Consumidor: Recibe las tareas de la cola, donde aplica el filtro a la particion, sube la parte procesada a el bucket y una vez hecho eso, le avisa al joiner que terminó.

- Joiner: Recibe por un endpoint que un consumidor terminó su tarea, va a redis y en el caso de que hayan terminado los consumidores de armar una imagen, las junta y arma la versión final de la imagen subiendola a el bucket.

# Requisitos de ejecución

Para esto se necesita tener instalado docker y docker-compose, el cual para usar la aplicacion deberan tener en cada carpeta de cliente, consumidor y joiner, un archivo de autenticacion para poder hacer uso del bucket que utiliza esta versión de Hit#1, el cual se proveera por correo electrónico, este archivo tiene que ir 3 veces en cada carpeta y tiene que ir adentro con una carpeta llamada "credentials", deberia quedar algo asi:

joiner/credentials/prueba-3fc1f-b8f34eda9b2e.json
cliente/credentials/prueba-3fc1f-b8f34eda9b2e.json
consumidor/credentials/prueba-3fc1f-b8f34eda9b2e.json

Luego de esto usar:

- docker-compose up --build



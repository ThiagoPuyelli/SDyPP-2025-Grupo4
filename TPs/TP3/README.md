# TP Nro 3 - Sistemas Distribuidos y Programacion Paralela

## Video presentacion
[Video](https://youtu.be/0WL-X8g0s3M)

### Integrantes:
* Puyelli, Thiago
* Herrneder, Matias

## Consigna
Trabajo Práctico Nº 3
Computación en la Nube (Kubernetes / RabbitMQ)
Fecha de Entrega: 11/05/25
(todos los puntos)

En la nube, tengo 2 patrones de trabajo.
Cloud Native (corre 100 % en la nube).
Híbrido (parte en equipos locales / Onpremise - Cloud)

Ahora, pensemos la nube como una extensión de la capacidad de cómputo de su/sus equipo/s. De esta manera, logramos aplicar el patrón de Cloud-Bursting. 

Por ejemplo, si tenemos una aplicación Cliente/Servidor (HTTP, gRPC, WSS) como por ejemplo filtro Sobel……..

#HIT 1 - El operador de Sobel es una máscara que, aplicada a una imagen, permite detectar (resaltar) bordes. Este operador es una operación matemática que, aplicada a cada pixel y teniendo en cuenta los píxeles que lo rodean, obtiene un nuevo valor (color) para ese pixel. Aplicando la operación a cada píxel, se obtiene una nueva imagen que resalta los bordes.
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

Bien, todo eso lo corrieron “distribuido” pero “centralizado” en su propia computadora. Sin embargo, la necesidad de procesamiento para aplicar un filtro sobre imágenes (que pueden ser muy grandes) puede generar que nos encontramos con un tope de capacidad de cómputo (más allá que lo hayamos construido de forma redundante), ¿Qué opciones tenemos?
Si es local, agregar más nodos de trabajo. Sin embargo, como discutimos, eso requiere tiempo (compra, ambientación, espacio e instalación de paquetes) y produce un Costo Fijo mínimo (más allá del variable por intensidad de uso). 

Hacer un offloading a la nube. La cual trabajaremos en este apartado. 

#HIT 2 - Sobel con offloading en la nube ;) para construir una base elástica (elástica):
Mismo objetivo de calcular sobel, pero ahora vamos a usar Terraform para construir nodos de trabajo cuando se requiera procesar tareas y eliminarlos al terminar. Recuerde que será necesario:
Instalar con #user_data las herramientas necesarias (java, docker, tools, docker).
Copiar ejecutable (jar, py, etc) o descargar imagen Docker (hub).
Poner a correr la aplicación e integrarse al cluster de trabajo.

El objetivo de este ejercicio es que ustedes puedan construir una arquitectura escalable (tipo 1, inicial) HÍBRIDA. Debe presentar el diagrama de arquitectura y comentar su decisión de desarrollar cada servicio y donde lo “coloca”.


#HIT 3 - Sobel contenerizado asincrónico y escalable (BASE DE TP FINAL) 

A diferencia del clúster anterior, la idea es que construya una infraestructura basada en la nube pero ahora con un enfoque diferente. 

Para ello, será necesario:

Desplegar con terraform un cluster de Kubernetes (GKE). 

Este será el manejador de todos los recursos que vayamos a desplegar. Es decir, va a alojar tanto los servicios de infraestructura (rabbitMQ y Redis) como los componentes de las aplicaciones que vamos a correr (frontend, backend, split, joiner, etc). Este clúster tiene que tener la siguiente configuración mínima
Un nodegroup para alojar los servicios de infraestructura (rabbitmq, redis, otros)
Un nodegroup compartido para las aplicaciones del sistema (front, back, split, joiner)
Máquinas virtuales (fuera del cluster)  que se encarguen de las tareas de procesamiento / cómputo intensivo. 


Construir los pipelines de despliegue de todos los servicios.

Pipeline 1: El que construye el Kubernetes. 
Pipeline 1.1: El que despliega los servicios (base datos - Redis, sistema de colas - RabbitMQ)
Pipeline 1.2-1.N: De cada aplicación desarrollada (frontend, backend, split, join)
Pipeline 2: Despliegue de máquinas virtuales para construir a los workers. Objetivo deseable: Que estas máquinas sean “dinámicas”.  



#HIT3 - Análisis de Desempeño Bajo Carga

Para evaluar el desempeño de la plataforma, se analizarán los tiempos de respuesta en diferentes escenarios de carga, modificando las siguientes variables:

Tamaño de los datos (V1): Se utilizarán tamaños representativos de carga, tales como 1 KB, 10 KB, 1 MB, 10 MB y 100 MB.
Cantidad de peticiones concurrentes (V2): Se simularán diferentes niveles de concurrencia.
Cantidad de workers (V3): Se ajustará el número de procesos o threads disponibles para manejar las peticiones.

⚡Disclaimer: Si quiere lograr usar más nodos, use nodos en distintas regiones 🙂

El objetivo es comprender cómo la plataforma responde al modificar estas variables, permitiendo identificar su capacidad de escalabilidad y posibles puntos de mejora. Los resultados se presentarán en una tabla que muestre los tiempos de respuesta obtenidos en cada combinación de las variables, proporcionando una visión clara de la evolución del desempeño bajo diferentes condiciones de carga.

## Consideraciones generales:
Cada ejercicio (Hit) tiene su propio readme con instrucciones y especificaciones puntuales del ejercicio, en rasgos generales, se empezó en el Hit#1 planteando una infraestructura que resolviera de forma distribuida el problema de aplicar un filtro sobel a una imagen, partiendo dicha imagen y paralelizando su procesamieno en distintas máquinas, a través de uso de docker.  
Luego, en el Hit#2, se avanzo a una estructura de cloud, donde se implementa todo el procesamiento en la nube (gcp), automatizando el despliegue con kubernetes.  
Finalmente, en el Hit#3, se implementa un sistema hibrido, donde si bien se sigue manteniendo gcp para el total del procesamiento, las tareas asociadas a workers y la aplicacion de los filtros, se realizan en maquinas virtuales (fuera del cluster de kubernetes), que no necesariamente tendrian que estar en gcp.

## Como utilizar la aplicación:
Para utilizar la aplicación, se debe realizar una petición HTTP GET con el formato: 
>http://{ip}/sobel?image_url={url_a_imagen_en_internet}&n_parts={cantidad_de_particiones_para_procesar}

Ejemplo:
```
http://34.138.218.127/sobel?image_url=https://imgs.search.brave.com/VyMDMutzQJTOYQZjnv9yhXb1NfyicOiLlexCkDe58KU/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zMS5z/aWduaWZpY2Fkb3Mu/Y29tL2ZvdG8vaW1h/Z2VuLWluaWNpby1j/a2UuanBnP2NsYXNz/PWFydGljbGU
```
```
http://34.138.218.127/sobel?image_url=https://imgs.search.brave.com/KpE6aao3PDrF0b5DYZhG4fYQMxSnrNSS43-VCt1n6LQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4z/LnBpeGVsY3V0LmFw/cC91cHNjYWxlX2Fm/dGVyXzJfNWY3N2Rh/YjkwYy5qcGc&n_parts=6
```

Para recuperar la imagen se usa el link que vuelve en la respuesta de la peticion, algo del estilo
>http://{ip}/get_image/{imagen_id}

Ejemplo:
```
http://34.138.218.127/get_image/8e55b9e8-9eb1-4c39-bd25-bf5a0e591164
```

## ATENCION
Por el momento solo se aceptan imagenes en formato .jpg
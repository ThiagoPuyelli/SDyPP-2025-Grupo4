Hit#7

Modifique el programa C y D, de manera tal de implementar un “sistema de inscripciones”, esto es, se define una ventana de tiempo fija de 1 MIN, coordinada por D, y los nodos C deben registrarse para participar de esa ventana, cuando un nodo C se registra a las 11:28:34 en D, el registro se hace efectivo para la próxima ventana de tiempo que corresponde a las 11:29. Cuando se alcanza las 11:29:00 el nodo D cierra las inscripciones y todo nodo C que se registre será anotado para la ventana de las 11:30, los nodos C que consulten las inscripciones activas solo pueden ver las inscripciones de la ventana actual, es decir, los nodos C no saben a priori cuales son sus pares para la próxima ventana de tiempo, solo saben los que están activos actualmente. Recuerde almacenar las inscripciones en un archivo de texto con formato JSON. Esto facilitará el seguimiento ordenado de las ejecuciones y asegurará la verificación de los resultados esperados.

Para simplificar el problema, imagine que D lleva dos registros, un listado de los nodos C activos en la ventana actual, y un registro de nodos C registrados para la siguiente ventana. Cada 60 segundos el nodo D mueve los registros de las inscripciones futuras a la presente y comienza a inscribir para la siguiente ronda.



# Creamos un volumen en docker

- Creamos un volumen en docker para almacenar los logs del ejercicio

docker volume create logs_volume

# Se crea una red

- Creamos una red para el ejercicio

sudo docker network create red_e

# Generamos las imagenes del programa

- Construimos las imagenes del programa en un docker-compose

sudo docker-compose build

# Iniciamos el registro "D"

- Al codigo por parametro se le proporciona su propia ip y se lo asigna a la red que creamos llamada red_d, ademas lo levantamos con el volumen de logs montado

sudo docker run -it --rm --network red_d --name registro_d -v logs_volume:/app/logs hit7-registro_d 0.0.0.0 5000

# Y los subsecuentes nodos "C"

- Al codigo por parametro se le proporciona la ip y puerto del registro_d, no le asignamos un nombre para poder correr el comando multiples veces

sudo docker run -it --rm --network red_d hit7-nodo_c registro_d 5000

# Accedemos a los logs

- Para poder ver el archivo de logs, creamos un contenedor temporal

sudo docker run --rm -v logs_volume:/app/logs busybox cat /app/logs/application.log


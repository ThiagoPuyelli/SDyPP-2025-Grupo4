Hit#6

Cree un programa D, el cual, actuará como un “Registro de contactos”. Para ello, en un array en ram, inicialmente vacío, este nodo D llevará un registro de los programas C que estén en ejecución.

Modifique el programa C de manera tal que reciba por parámetros únicamente la ip y el puerto del programa D. C debe iniciar la escucha en un puerto aleatorio y debe comunicarse con D para informarle su ip y su puerto aleatorio donde está escuchando. D le debe responder con  las ips y puertos de los otros nodos C que estén corriendo, haga que C se conecte a cada uno de ellos y envíe el saludo.

Es decir, el objetivo de este HIT es incorporar un nuevo tipo de nodo (D) que actúe como registro de contactos para que al iniciar cada nodo C no tenga que indicar las ips de sus pares. Esto debe funcionar con múltiples instancias de C, no solo con 2.


# Se crea una red

- Creamos una red para el ejercicio

sudo docker network create red_d

# Generamos las imagenes del programa

- Construimos las imagenes del programa en un docker-compose

sudo docker-compose build

# Iniciamos el registro "D"

- Al codigo por parametro se le proporciona su propia ip y se lo asigna a la red que creamos llamada red_d

sudo docker run -it --rm --network red_d --name registro_d hit6_registro_d 0.0.0.0 5000

# Y los subsecuentes nodos "C"

- Al codigo por parametro se le proporciona la ip y puerto del registro_d, no le asignamos un nombre para poder correr el comando multiples veces

sudo docker run -it --rm --network red_d hit6_nodo_c registro_d 5000

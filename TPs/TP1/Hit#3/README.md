Hit#3

Modifique el código de B para que si el proceso A cierra la conexión (por ejemplo matando el proceso) siga funcionando.

Para inicializar el ejercicio:

# Generamos las imagenes del servidor y cliente

- Nos posicionamos en la carpeta Hit#1 y lanzamos estos comandos (el nombre mi-servidor o mi-cliente se puede cambiar)

sudo docker build -t mi-servidor -f server/server.dockerfile server/
sudo docker build -t mi-cliente -f client/client.dockerfile client/

# Iniciar el servidor

sudo docker run -d --name servidor -p 12345:12345 mi-servidor

- Con el siguiente comando podemos ver lo que devuelve en consola el servidor, en si no deberia mostrar nada de momento

sudo docker logs servidor

# Vemos la ip del servidor

- De esta manera obtenemos la ip del servidor

sudo docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' servidor

- En mi caso me devolvio 172.17.0.3

# Iniciar cliente

sudo docker run -it --rm mi-cliente python client.py 172.17.0.3

- Inciamos el cliente con la ip brindada por el comando anterior, en si el programa deberia dejar enviar un mensaje cualquiera y devolver un saludo, luego desde el servidor se puede visualizar el log mediante el comando sudo docker logs servidor
Hit#5

Modifique el programa C de manera tal que los mensajes se envíen en formato JSON, serializar y deserializar los mismos al enviar/recibir. 

# Generamos las imagenes del programa

- Con un solo comando se construye la imagen, en este caso le pondremos programac

sudo docker build -t programac .

# Se crea una red

- En este caso optamos por crear una red de docker con este comando, el cual optamos por ponerle redC

sudo docker network create redC

# Luego iniciamos los contenedores

- Al codigo por parametro se le proporciona la ip destino y origen y luego lo mismo con los puertos por lo tanto a la hora de la ejecucion se les especifica el cual asignamos 5000 a uno y 6000 al otro, como tambien se crean con el nombre de servidor1 y servidor2, y ademas se asigna la red que creamos llamado redC

sudo docker run -it --rm --network redC --name servidor1 programac 0.0.0.0 5000 servidor2 6000
sudo docker run -it --rm --network redC --name servidor2 programac 0.0.0.0 6000 servidor1 5000





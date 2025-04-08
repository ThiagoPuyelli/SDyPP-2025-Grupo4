# Instrucciones para ejecutar:
### Correr por unica vez:
Crea una red global en docker para poder comunicarse con la tarea
> docker network create --driver bridge --attachable tarea-net

### Levantar Servidor
> docker-compose up servidor

### Correr cliente de consola (tambien se puede realizar con curl)
> docker-compose run --rm cliente

### Ejemplo en powershell:  
> Invoke-RestMethod -Uri "http://localhost:5000/getRemoteTask" -Method POST -Headers @{ "Content-Type" = "application/json" } -Body '{ "credenciales": null, "imagen": "matiasherrneder/tarea-cliente:latest", "tarea": "sumar", "parametros": { "x": 2, "y": 3 } }'

### Ejemplo usando curl:
> curl -X POST http://localhost:5000/getRemoteTask \
> -H "Content-Type: application/json" \
> -d '{
>   "credenciales": null,
>   "imagen": "matiasherrneder/tarea-cliente:latest",
>   "tarea": "sumar",
>   "parametros": {
>     "x": 2,
>     "y": 3
>   }
> }'

# Credenciales
En caso de no usar el cliente se necesitara encriptar las credenciales antes de ponerlas en el json

### Clave publica del servidor:
> -----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwQ61E7I5hKBqbsqt48sY
hmkEdFZGxyPUMPSOd08nlBLLcSCnHJzEAX7TtdSgj/rX+pEzOs/tsPPAdraNRSVk
iOgZ2vd7+cQkxmQvehZbxzNExwN+I81efDjlzOM0p3Oucg8ZmEbJJOZ7ohngDPsI
EGqB8vo7nhmq93b1eOuaOgUDsoqfN4ubkW3H9SD3LIgdBDDVn+xptWoF0LKSS9k9
8Wb2j3HHdNHdXmbMx37WEEaRcSxBGow6tNPl6SmsbbWQQWEhnbSGRKMRyBnOmUfr
mh8M5pYIU3poK6zmIIKMj0GQmTHm5ZOnwAgUGPR+cg71l8R9Gru4fIBhhokmRPpt
wwIDAQAB
-----END PUBLIC KEY-----


### Usuario en Docker Hub para pruebas (repo privado): 
>* user: testsdunlu
>* pass: estoesuntest
>* imagen: testsdunlu/cliente_privado:latest
>* funcion: medir_respuesta_http(url, intentos, intervalo)


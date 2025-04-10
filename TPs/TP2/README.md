# FUNCIONAMIENTO DE LA APLICACION:
#### Este repositorio contiene varias partes:
**Servidor:** Recibe las peticiones del usuario, se encarga de hacer pull de las tareas, ejecutarlas y devolver una respuesta.  
**Tarea generica:** Es una imagen transparente para el usuario, que implementa la logica de comunicacion con el servidor, sobre esta los usuarios crean sus propias tareas.  
**Imagenes de ejemplo:** Son ejemplos de tareas que crearian los usuarios del servidor, implementan una estructura muy simple, con un modulo tareas.py donde escriben nada mas que sus funciones y un dockerfile que construye la tarea a partir de la generica.  
**Cliente de prueba:** Un cliente para probar ingresando los datos por consola, tiene integrada una prueba basica, se puede ejecutar tambien con una peticion curl o Invoke-RestMethod, como se ejemplifica mas adelante.

## Imagenes disponibles en Docker Hub
- **Servidor:** matiasherrneder/servidor-tareas:latest
- **Tarea generica:** matiasherrneder/servidor-tareas:latest

## Instrucciones para ejecutar el servidor:
### Crear una red global en docker para comunicarse con la tarea
Correr por unica vez:
> docker network create --driver bridge --attachable tarea-net

### Levantar Servidor
> docker-compose up servidor

## Como correr pruebas:
Para correr las tareas no hace falta descargarlas, solo conectarse al servidor y enviarle los parametros correctos, en un json con el formato:
>credenciales: str | None *-(credenciales encriptadas o null)*  
>imagen: str *-(nombre de la funcion)*  
>tarea: str *-(nombre de la imagen en Docker Hub)*  
>parametros: dict | None *-(parametros en formato clave valor o null)*  

** Las credenciales deben ser encriptadas como un diccionario del siguiente formato: (para esto se puede usar el cliente)
>usuario: str  
contrasena: str

### Correr cliente de consola
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

## Autenticacion y credenciales
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
>* credenciales encriptadas: "AdpTJzwJHImRsEaGUWaC6ENVBm9rLkIi69xVgGeU2/M+Yz0U2q0vO9y/N8VLgDRDvaE8ar2bda7nC7v/xIzzcOerAuXBH6dXUoyf7xkZIeoPXvlhBbmzJI4yo0fJHNDXcPUdHVbV4Tw/AEyplnWt4TYfg4f6ZkBh3LdsMahxOenQauK5t2GbuLtMkMBqNMrpM4hrbQ7tJtdptZcvkD2pUI/+/noBusB0m2VQQxkSmWk4d+DEHhcbseexaxZheo0U7Mh43glOIXPkBqOfMUWXxdSOnTpajUJFeZlukcY/X33rPI0XZrBunb2V6H+H8rMXCpxDQ8IPwIrxfb2fPiDHhQ=="

### Ejemplo en poweshell con imagen privada en Docker Hub y credenciales cifradas

>Invoke-RestMethod -Uri "http://localhost:5000/getRemoteTask" -Method POST -Headers @{ "Content-Type" = "application/json" } -Body '{ "credenciales": "AdpTJzwJHImRsEaGUWaC6ENVBm9rLkIi69xVgGeU2/M+Yz0U2q0vO9y/N8VLgDRDvaE8ar2bda7nC7v/xIzzcOerAuXBH6dXUoyf7xkZIeoPXvlhBbmzJI4yo0fJHNDXcPUdHVbV4Tw/AEyplnWt4TYfg4f6ZkBh3LdsMahxOenQauK5t2GbuLtMkMBqNMrpM4hrbQ7tJtdptZcvkD2pUI/+/noBusB0m2VQQxkSmWk4d+DEHhcbseexaxZheo0U7Mh43glOIXPkBqOfMUWXxdSOnTpajUJFeZlukcY/X33rPI0XZrBunb2V6H+H8rMXCpxDQ8IPwIrxfb2fPiDHhQ==", "imagen": "testsdunlu/cliente_privado:latest", "tarea": "medir_respuesta_http", "parametros": { "url": "https://www.google.com", "intentos": 10, "intervalo": 0.2 } }'
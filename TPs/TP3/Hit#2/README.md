# Hit 2 Sobel con offloading en la nube ;) para construir una base elástica (elástica):
Mismo objetivo de calcular sobel, pero ahora vamos a usar Terraform para construir nodos de trabajo cuando se requiera procesar tareas y eliminarlos al terminar. Recuerde que será necesario:
Instalar con #user_data las herramientas necesarias (java, docker, tools, docker).
Copiar ejecutable (jar, py, etc) o descargar imagen Docker (hub).
Poner a correr la aplicación e integrarse al cluster de trabajo.

El objetivo de este ejercicio es que ustedes puedan construir una arquitectura escalable (tipo 1, inicial) HÍBRIDA. Debe presentar el diagrama de arquitectura y comentar su decisión de desarrollar cada servicio y donde lo “coloca”.

# Resolucion

Para resolver este enunciado, utilizamos terraform y kubernetes siguiendo el diagrama.png, donde utilizamos goole cloud para que se mantenga funcionando en internet, en si el funcionamiento de cada parte esta explicado en el Hit#1, luego aca dejamos los comandos para poder hacer uso del proyecto terraform.

# IMPORTANTE, siempre correr al finalizar
>terraform destroy

## Uso de configuracion:

### Descargar gpc CLI
> https://cloud.google.com/sdk/docs/install

### Logearse con gcp e instalar plugins
```
gcloud auth login
gcloud components install gke-gcloud-auth-plugin
```

### Crear una cuenta de servicio para kubernetes
* Crear una cuenta de servicio (con permisos o roles de: Kubernetes Engine Admin, Viewer), descargar la clave y ponerla en la carpeta credentials con el nombre auth.json

##
### Cambios de posibles de configuracion:
* Modificar el archivo variables.tf para especificar los datos del proyecto

## Para cargar en docker hub las imagenes:
```
docker tag TU_IMAGEN TU_USUARIO_DOCKER/TU_TAG:latest
docker push TU_USUARIO_DOCKER/TU_TAG:latest
```

## Comandos para levantar el cluster:

### Inicia terraform
```
terraform init
```

### Muestra los cambios a incorporar con apply (o errores si hay)
```
terraform plan
```

### Aplicar la configuracion del cluster
```
terraform apply
```

### Conectarse al cluster creado
```
gcloud container clusters get-credentials sdeath-cluster --region us-east1-b --project sd-2025-459518
```

### Verificar conexion
```
kubectl get nodes
```

### Despliegue de las aplicaciones
```
kubectl apply -f ../manifests/.
```

## Recuperar ip del servidor
```
kubectl get svc cliente
```

## Aplicar a una imagen en internet
Con la ip del servidor:
```
http://34.75.184.36/sobel?image_url=https://imgs.search.brave.com/VyMDMutzQJTOYQZjnv9yhXb1NfyicOiLlexCkDe58KU/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zMS5z/aWduaWZpY2Fkb3Mu/Y29tL2ZvdG8vaW1h/Z2VuLWluaWNpby1j/a2UuanBnP2NsYXNz/PWFydGljbGU

http://34.75.184.36/sobel?image_url=https://imgs.search.brave.com/KpE6aao3PDrF0b5DYZhG4fYQMxSnrNSS43-VCt1n6LQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4z/LnBpeGVsY3V0LmFw/cC91cHNjYWxlX2Fm/dGVyXzJfNWY3N2Rh/YjkwYy5qcGc&n_parts=6
```
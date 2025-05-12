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
```
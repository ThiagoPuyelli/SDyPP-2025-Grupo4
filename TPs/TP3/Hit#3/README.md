# HIT 3 - Sobel contenerizado asincrónico y escalable (BASE DE TP FINAL) 

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

# Resolucion

Dejamos los mismos comandos que el Hit 2 ya que es muy similar, en cuanto a este problema decidimos en terraform utilizar maquinas virtuales, el cual funcionan de forma externa al cluster, permitiendo asi que los workers no necesariamente pertenezcan en si mismo al cluster

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
gcloud container clusters get-credentials tp3hit3-cluster --region us-east1-b --project sd-hit3
```

### Verificar conexion
```
kubectl get nodes
```

### Despliegue de las aplicaciones
```
kubectl apply -f ../manifests/.
```

## Editar el archivo terraform/13-vms.tf
Obtener las ips de los servicios del joiner y rabbit y ponerlas en el archivo, luego volver a desplegar el archivo
```
kubectl get svc
```

## Recuperar ip del servidor
```
kubectl get svc cliente
```

## Aplicar a una imagen en internet
Con la ip del servidor:
```
http://34.138.218.127/sobel?image_url=https://imgs.search.brave.com/VyMDMutzQJTOYQZjnv9yhXb1NfyicOiLlexCkDe58KU/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zMS5z/aWduaWZpY2Fkb3Mu/Y29tL2ZvdG8vaW1h/Z2VuLWluaWNpby1j/a2UuanBnP2NsYXNz/PWFydGljbGU

http://34.138.218.127/sobel?image_url=https://imgs.search.brave.com/KpE6aao3PDrF0b5DYZhG4fYQMxSnrNSS43-VCt1n6LQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4z/LnBpeGVsY3V0LmFw/cC91cHNjYWxlX2Fm/dGVyXzJfNWY3N2Rh/YjkwYy5qcGc&n_parts=6

```

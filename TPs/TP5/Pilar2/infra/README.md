# IMPORTANTE, no dejar el cluster prendido innecesariamente
```
terraform destroy
```
##

## Uso de configuracion:

### Descargar gpc CLI
* https://cloud.google.com/sdk/docs/install

### Logearse con gcp e instalar plugins
```
gcloud auth login
gcloud components install gke-gcloud-auth-plugin
```

### Crear una cuenta de servicio para kubernetes
* Crear una cuenta de servicio (con permisos o roles de: Kubernetes Engine Admin, Viewer), descargar la clave y ponerla en la carpeta credentials con el nombre auth.json

## Imagenes cargadas en Docker Hub:
```
docker tag matiasherrneder/mipla-backend:latest
docker push matiasherrneder/mipla-backend:latest
docker tag TU_IMAGEN matiasherrneder/mipla-frontend:latest
docker push matiasherrneder/mipla-frontend:latest
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
gcloud container clusters get-credentials $(terraform output -raw cluster_name) --region $(terraform output -raw zone) --project $(terraform output -raw project_id)
```

### Verificar conexion
```
kubectl get nodes
```

### Configurar Vault y cargar secretos desde `.env`
1. Crear el archivo `vault/.env` copiando `vault/.env.example` y completar `RABBIT_USER`, `RABBIT_PASS`, `REDIS_DB`.
2. Desplegar Vault (namespace `secret`) y cargar los valores en `secret/data/blockchain`:
   ```
   ./vault/deploy.sh
   ```
   Podés pasar otro `.env` como primer argumento si no está en `vault/.env`.
3. Verificar que el Job de bootstrap terminó:
   ```
   kubectl -n secret get jobs
   ```
   Debe aparecer `vault-bootstrap` en estado `Complete`.

### Observabilidad (Loki + Promtail + Grafana)
1. Ir a `observability` y ejecutar:
   ```
   ./observability/install.sh
   ```
   Instala el chart `loki-stack` en el namespace `observability` con Loki+Promtail+Grafana, PVCs y dashboard de logs.
2. Acceso a Grafana:
   ```
   kubectl -n observability get svc loki-stack-grafana
   ```
   - Si es LoadBalancer, usar la IP externa.
   - O `kubectl -n observability port-forward svc/loki-stack-grafana 3000:80`.
   Usuario/clave por defecto: `admin` / `admin`.

### Despliegue de los manifiestos
En el directorio /manifests

```
kubectl apply -f . --recursive
```

### Verificar pods
```
kubectl get pods -n blockchain
```

### Conseguir IP del frontend para conectarse
```
kubectl get ingress -n blockchain
```
Obtenemos algo como:

| NAME | CLASS | HOSTS | ADDRESS | PORTS | AGE
|-|-|-|-|-|-
| coordinador-ingress | <none> | * | 35.244.137.250 | 80 | 93s

`IP: 35.244.137.250`

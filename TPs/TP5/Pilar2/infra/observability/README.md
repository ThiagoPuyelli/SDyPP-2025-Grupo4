# Observabilidad (Loki + Promtail + Grafana)

Stack listo para levantar con Helm en el namespace `observability`, con persistencia básica y dashboards precargados para Loki.

## Despliegue rápido
```bash
# Desde infra/observability
./install.sh
```
Verás un warning porque el chart `loki-stack` está deprecado; sigue funcionando, pero a futuro conviene migrar a `grafana/loki` + `grafana/promtail` + `grafana/grafana` por separado.
El script:
- Crea el namespace `observability`.
- Agrega el repo de Helm de Grafana.
- Instala/actualiza el release `loki-stack` con los valores de `values-loki-stack.yaml` (Loki, Promtail y Grafana).
- Lee `GRAFANA_USER`/`GRAFANA_PASS` desde Vault (`secret/blockchain`) y crea el Secret `grafana-admin` en `observability` (necesita Vault corriendo).
- Espera el rollout de Grafana y del StatefulSet de Loki (`loki-stack`). Si el nombre cambia, ajustar en `install.sh`.

Acceso a Grafana:
- Usuario/clave por defecto: `admin` / `admin`.
- Servicio: `kubectl -n observability get svc loki-stack-grafana`.
  - Si es `LoadBalancer`, usar la IP externa.
  - O hacer port-forward: `kubectl -n observability port-forward svc/loki-stack-grafana 3000:80`.

## Qué queda configurado
- Loki con almacenamiento en PVC (`10Gi`, storageClass `standard-rwo`).
- Promtail con scrape extra para namespaces `blockchain` y `secret`.
- Grafana con persistencia (`5Gi`), datasource Loki preconfigurado, dashboard de logs (ID 13639) y credenciales tomadas de Vault (`grafana-admin` Secret).
- Servicio de Grafana tipo `LoadBalancer` (editar en `values-loki-stack.yaml` si querés ClusterIP/NodePort).

## Dashboards propios (persistidos en el repo)
- Exportá tu dashboard desde Grafana (JSON) y guardalo en `dashboards/` (por ejemplo `dashboards/mi-dashboard.json`).
- Al correr `./install.sh`, se crea/actualiza el ConfigMap `grafana-dashboards-custom` con todos los JSON de esa carpeta y Grafana los monta en `/var/lib/grafana/dashboards/default`.
- Así, aunque borres el cluster o el PVC, al reinstalar el chart se cargan los dashboards guardados en el repo.

## Notas sobre datos y destrucción del cluster
- Los datos viven en PVC locales al cluster; si hacés `terraform destroy` del cluster, se pierden. Para retener logs/dashboards tras recrear el cluster, necesitarías apuntar a almacenamiento externo (p. ej., bucket + configuración de Loki para `boltdb-shipper` con GCS/S3) o marcar los PV con reclaim `Retain` y reusarlos.
- Si querés limpiar solo el stack: `helm uninstall loki-stack -n observability` y luego borrar el namespace si no lo usás.

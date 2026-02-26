# Vault en Kubernetes (namespace `secret`)

Este flujo levanta Vault en modo `server` (no dev), con almacenamiento persistente en PVC y auto-unseal con GCP KMS. También carga secretos desde `.env` y configura auth Kubernetes.

## Pasos rápidos
1. Completar secretos y parámetros:
   - Copiar `TPs/TP5/Pilar2/infra/vault/.env.example` a `.env`.
   - Definir al menos: `RABBIT_USER`, `RABBIT_PASS`, `REDIS_DB`, `REDIS_PASSWORD`, `POOL_PK`.
   - Si querés forzar KMS explícito: `GCP_PROJECT_ID`, `KMS_REGION`, `KMS_KEY_RING`, `KMS_CRYPTO_KEY`.
2. Desplegar y cargar Vault (usa el `.env` anterior por defecto, podés pasar otro como primer argumento):
   ```bash
   TPs/TP5/Pilar2/infra/vault/deploy.sh
   # o
   TPs/TP5/Pilar2/infra/vault/deploy.sh /ruta/a/.env
   ```
   El script:
   - Crea/reutiliza KeyRing y CryptoKey de Cloud KMS para auto-unseal.
   - Otorga `roles/cloudkms.cryptoKeyEncrypterDecrypter` al service account de nodos (por default `kubernetes@<project>.iam.gserviceaccount.com`).
   - Crea el namespace `secret` y cuentas de servicio necesarias.
   - Genera `ConfigMap/vault-config` con `server.hcl` (storage persistente + `seal "gcpckms"`).
   - Despliega Vault como StatefulSet (`vault-0`) con PVC.
   - Si Vault no estaba inicializado, corre `vault operator init` y guarda el token admin en `Secret/vault-admin`.
   - Crea un Secret temporal `vault-env` desde `.env`.
   - Ejecuta `Job/vault-bootstrap` que:
     - Habilita `kv-v2` en `secret/`.
     - Configura auth Kubernetes con roles `blockchain-role` y `pool-role`.
     - Escribe `secret/blockchain` y `secret/pool`.
   - Elimina `vault-env` al final.
3. Desplegar el resto de los manifests (después de que el Job termine):
   ```bash
   kubectl apply -f TPs/TP5/Pilar2/infra/manifests --recursive
   ```

## Detalles útiles
- Vault queda accesible como `vault.secret.svc:8200` (sin TLS, pensado para uso interno del cluster/laboratorio).
- El estado de Vault persiste en el PVC del StatefulSet: un reinicio del pod no borra secretos.
- El unseal es automático usando una CryptoKey de Cloud KMS (creada/reusada por `deploy.sh`).
- Si cambiás secretos en `.env`, podés rerun `deploy.sh` para reescribirlos en Vault.
- Para limpiar solo el bootstrap job:
  ```bash
  kubectl -n secret delete job vault-bootstrap
  ```
- Si hacés `terraform destroy`, se borra el cluster y sus PVC. Después de recrear con `terraform apply`, `deploy-all.sh` vuelve a levantar todo y recarga secretos desde `.env`.

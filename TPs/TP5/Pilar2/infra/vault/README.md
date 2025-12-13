# Vault en Kubernetes (namespace `secret`)

Este flujo levanta un Vault dev dentro del cluster, carga los secretos definidos en un `.env` y prepara la autenticación vía Kubernetes para que los pods los lean.

## Pasos rápidos
1. Completar los secretos: copiar `TPs/TP5/Pilar2/infra/vault/.env.example` a `.env` y setear `RABBIT_USER`, `RABBIT_PASS`, `REDIS_DB`.
2. Desplegar y cargar Vault (usa el `.env` anterior por defecto, podés pasar otro como primer argumento):
   ```bash
   TPs/TP5/Pilar2/infra/vault/deploy.sh
   # o
   TPs/TP5/Pilar2/infra/vault/deploy.sh /ruta/a/.env
   ```
   El script:
   - Crea el namespace `secret` y cuentas de servicio necesarias.
   - Despliega Vault en modo dev (`VAULT_DEV_ROOT_TOKEN_ID=root`).
   - Crea el Secret `vault-env` con los valores del `.env`.
   - Ejecuta el Job `vault-bootstrap` que:
     - Habilita `kv-v2` en `secret/`.
     - Configura auth Kubernetes con rol `blockchain-role` (service accounts `coordinador-sa` y `rabbitmq-sa` del namespace `blockchain`).
     - Escribe `secret/blockchain` con los valores del `.env`.
3. Desplegar el resto de los manifests (después de que el Job termine):
   ```bash
   kubectl apply -f TPs/TP5/Pilar2/infra/manifests --recursive
   ```

## Detalles útiles
- Vault queda accesible como `vault.secret.svc:8200` (sin TLS, solo para uso interno del cluster/demo).
- Los secretos se guardan en `secret/data/blockchain` con las claves `RABBIT_USER`, `RABBIT_PASS`, `REDIS_DB`.
- Si cambiás el `.env`, volvés a correr `deploy.sh`: recrea el Secret temporal y reejecuta el Job de bootstrap.
- Para limpiar solo el bootstrap job y el Secret temporal:
  ```bash
  kubectl -n secret delete job vault-bootstrap
  kubectl -n secret delete secret vault-env
  ```
- No usar este setup en producción: usa Vault en modo dev y sin TLS para simplificar el laboratorio.

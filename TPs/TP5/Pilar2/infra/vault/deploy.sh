#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_ROOT="${INFRA_ROOT}/manifests"
VAULT_MANIFESTS="${MANIFEST_ROOT}/vault"

ENV_FILE="${1:-${SCRIPT_DIR}/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "No se encontró el archivo de variables (${ENV_FILE}). Copiá .env.example y completalo." >&2
  exit 1
fi

# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

required_env_vars=("RABBIT_USER" "RABBIT_PASS" "REDIS_DB" "POOL_PK")
for required_var in "${required_env_vars[@]}"; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "La variable ${required_var} es obligatoria en ${ENV_FILE}." >&2
    exit 1
  fi
done

if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
  provider_id="$(kubectl get nodes -o jsonpath='{.items[0].spec.providerID}' 2>/dev/null || true)"
  if [[ "${provider_id}" == gce://* ]]; then
    GCP_PROJECT_ID="$(echo "${provider_id}" | cut -d/ -f3)"
  fi
fi

KMS_REGION="${KMS_REGION:-us-east1}"
KMS_KEY_RING="${KMS_KEY_RING:-vault-keyring}"
KMS_CRYPTO_KEY="${KMS_CRYPTO_KEY:-vault-unseal-key}"
KMS_AUTH_SERVICE_ACCOUNT="${KMS_AUTH_SERVICE_ACCOUNT:-kubernetes@${GCP_PROJECT_ID}.iam.gserviceaccount.com}"

if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
  echo "No se pudo inferir GCP_PROJECT_ID. Definilo en ${ENV_FILE}." >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI es obligatorio para crear/reusar la key de auto-unseal." >&2
  exit 1
fi

echo "Verificando recursos KMS para auto-unseal..."
if ! gcloud kms keyrings describe "${KMS_KEY_RING}" \
  --location="${KMS_REGION}" \
  --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
  gcloud kms keyrings create "${KMS_KEY_RING}" \
    --location="${KMS_REGION}" \
    --project="${GCP_PROJECT_ID}" >/dev/null
fi

if ! gcloud kms keys describe "${KMS_CRYPTO_KEY}" \
  --keyring="${KMS_KEY_RING}" \
  --location="${KMS_REGION}" \
  --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
  gcloud kms keys create "${KMS_CRYPTO_KEY}" \
    --keyring="${KMS_KEY_RING}" \
    --location="${KMS_REGION}" \
    --purpose="encryption" \
    --project="${GCP_PROJECT_ID}" >/dev/null
fi

gcloud kms keys add-iam-policy-binding "${KMS_CRYPTO_KEY}" \
  --keyring="${KMS_KEY_RING}" \
  --location="${KMS_REGION}" \
  --project="${GCP_PROJECT_ID}" \
  --member="serviceAccount:${KMS_AUTH_SERVICE_ACCOUNT}" \
  --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" >/dev/null

echo "Aplicando namespaces y cuentas de servicio..."
kubectl apply -f "${MANIFEST_ROOT}/namespace-coordinador.yaml"
kubectl apply -f "${MANIFEST_ROOT}/namespace-pool.yaml"
kubectl apply -f "${VAULT_MANIFESTS}/namespace.yaml"
kubectl apply -f "${VAULT_MANIFESTS}/blockchain-serviceaccounts.yaml"
kubectl apply -f "${VAULT_MANIFESTS}/pool-serviceaccount.yaml"

echo "Generando configuración de Vault (server mode + auto-unseal KMS)..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: vault-config
  namespace: secret
data:
  server.hcl: |
    ui = true
    disable_mlock = true

    listener "tcp" {
      address         = "0.0.0.0:8200"
      cluster_address = "0.0.0.0:8201"
      tls_disable     = 1
    }

    storage "file" {
      path = "/vault/data"
    }

    seal "gcpckms" {
      project    = "${GCP_PROJECT_ID}"
      region     = "${KMS_REGION}"
      key_ring   = "${KMS_KEY_RING}"
      crypto_key = "${KMS_CRYPTO_KEY}"
    }

    api_addr = "http://vault.secret.svc:8200"
    cluster_addr = "http://vault-0.vault-internal.secret.svc:8201"
EOF

echo "Desplegando Vault..."
kubectl apply -f "${VAULT_MANIFESTS}/vault.yaml"
kubectl -n secret rollout status statefulset/vault --timeout=300s

echo "Verificando estado de inicialización..."
STATUS_JSON=""
for _ in $(seq 1 60); do
  STATUS_JSON="$(kubectl -n secret exec vault-0 -- sh -c 'VAULT_ADDR=http://127.0.0.1:8200 vault status -format=json' 2>/dev/null || true)"
  if [[ "${STATUS_JSON}" == *"initialized"* ]]; then
    break
  fi
  sleep 2
done

if [[ "${STATUS_JSON}" != *"initialized"* ]]; then
  echo "No se pudo obtener el estado de Vault." >&2
  exit 1
fi

if echo "${STATUS_JSON}" | grep -q '"initialized":[[:space:]]*false'; then
  echo "Vault no estaba inicializado. Ejecutando init..."
  INIT_JSON="$(kubectl -n secret exec vault-0 -- sh -c 'VAULT_ADDR=http://127.0.0.1:8200 vault operator init -key-shares=1 -key-threshold=1 -format=json')"
  INIT_JSON_SINGLE_LINE="$(echo "${INIT_JSON}" | tr -d '\n')"
  ROOT_TOKEN="$(echo "${INIT_JSON_SINGLE_LINE}" | sed -n 's/.*"root_token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  RECOVERY_KEY="$(echo "${INIT_JSON_SINGLE_LINE}" | sed -n 's/.*"recovery_keys_b64"[[:space:]]*:[[:space:]]*\["\([^"]*\)".*/\1/p')"

  if [[ -z "${ROOT_TOKEN}" ]]; then
    echo "No se pudo extraer el root token del init de Vault." >&2
    exit 1
  fi

  kubectl -n secret create secret generic vault-admin \
    --from-literal=token="${ROOT_TOKEN}" \
    --from-literal=recovery_key="${RECOVERY_KEY}" \
    --dry-run=client -o yaml | kubectl apply -f -

  echo "Vault inicializado. Token admin guardado en secret/vault-admin."
else
  echo "Vault ya estaba inicializado."
  if ! kubectl -n secret get secret vault-admin >/dev/null 2>&1; then
    echo "Vault está inicializado pero falta secret/vault-admin con el token." >&2
    echo "Crealo manualmente para poder correr el bootstrap." >&2
    exit 1
  fi
fi

echo "Publicando variables del .env como Secret temporal..."
kubectl -n secret create secret generic vault-env \
  --from-env-file="${ENV_FILE}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Ejecutando bootstrap para cargar secretos y roles..."
kubectl -n secret delete job vault-bootstrap --ignore-not-found
kubectl apply -f "${VAULT_MANIFESTS}/bootstrap-job.yaml"
kubectl -n secret wait --for=condition=complete job/vault-bootstrap --timeout=300s

echo "Eliminando Secret temporal vault-env..."
kubectl -n secret delete secret vault-env --ignore-not-found

echo "Vault desplegado y secretos cargados en secret/data/blockchain y secret/data/pool."
echo "Ahora podés aplicar el resto de los manifiestos con:"
echo "  kubectl apply -f ${MANIFEST_ROOT} --recursive"

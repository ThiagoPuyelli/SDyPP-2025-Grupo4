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

echo "Aplicando namespaces y cuentas de servicio..."
kubectl apply -f "${MANIFEST_ROOT}/namespace-coordinador.yaml"
kubectl apply -f "${VAULT_MANIFESTS}/namespace.yaml"
kubectl apply -f "${VAULT_MANIFESTS}/blockchain-serviceaccounts.yaml"

echo "Desplegando Vault..."
kubectl apply -f "${VAULT_MANIFESTS}/vault.yaml"
kubectl -n secret rollout status deploy/vault --timeout=300s

echo "Publicando variables del .env como Secret temporal..."
kubectl -n secret create secret generic vault-env \
  --from-env-file="${ENV_FILE}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Ejecutando bootstrap para cargar secretos y roles..."
kubectl -n secret delete job vault-bootstrap --ignore-not-found
kubectl apply -f "${VAULT_MANIFESTS}/bootstrap-job.yaml"
kubectl -n secret wait --for=condition=complete job/vault-bootstrap --timeout=300s

echo "Vault desplegado y secretos cargados en secret/data/blockchain."
echo "Ahora podés aplicar el resto de los manifiestos con:"
echo "  kubectl apply -f ${MANIFEST_ROOT} --recursive"

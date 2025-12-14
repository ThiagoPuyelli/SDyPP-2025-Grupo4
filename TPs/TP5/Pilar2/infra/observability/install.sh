#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES_FILE="${SCRIPT_DIR}/values-loki-stack.yaml"
NAMESPACE="observability"
RELEASE="loki-stack"
ENV_FILE="${SCRIPT_DIR}/../vault/.env"
DASHBOARDS_DIR="${SCRIPT_DIR}/dashboards"

echo "Creando namespace ${NAMESPACE}..."
kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"

echo "Agregando repo de Helm grafana..."
helm repo add grafana https://grafana.github.io/helm-charts >/dev/null
helm repo update >/dev/null

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "No se encontró el archivo ${ENV_FILE}. Copiá ../vault/.env.example a ../vault/.env y completalo." >&2
  exit 1
fi

# Carga GRAFANA_USER y GRAFANA_PASS desde el .env de Vault
set -a
source "${ENV_FILE}"
set +a

if [[ -z "${GRAFANA_USER:-}" || -z "${GRAFANA_PASS:-}" ]]; then
  echo "GRAFANA_USER o GRAFANA_PASS están vacíos en ${ENV_FILE}." >&2
  exit 1
fi

kubectl -n "${NAMESPACE}" create secret generic grafana-admin \
  --from-literal=admin-user="${GRAFANA_USER}" \
  --from-literal=admin-password="${GRAFANA_PASS}" \
  --dry-run=client -o yaml | kubectl apply -f -

if compgen -G "${DASHBOARDS_DIR}/*.json" >/dev/null; then
  echo "Publicando dashboards locales en ConfigMap grafana-dashboards-custom..."
  kubectl -n "${NAMESPACE}" create configmap grafana-dashboards-custom \
    --from-file="${DASHBOARDS_DIR}" \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "No se encontraron dashboards locales en ${DASHBOARDS_DIR}; se usarán solo los incluidos por el chart."
fi

echo "Instalando/actualizando ${RELEASE} en ${NAMESPACE}..."
helm upgrade --install "${RELEASE}" grafana/loki-stack \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  -f "${VALUES_FILE}"

echo "Esperando a que Grafana y Loki estén listos..."
kubectl -n "${NAMESPACE}" rollout status deploy/"${RELEASE}"-grafana --timeout=300s
# El chart crea el StatefulSet con el nombre del release (sin sufijo -loki).
kubectl -n "${NAMESPACE}" rollout status statefulset/"${RELEASE}" --timeout=300s || true

echo "Listo. Accedé a Grafana (usuario/pass tomados de ${ENV_FILE}) vía:"
echo "  kubectl -n ${NAMESPACE} get svc ${RELEASE}-grafana"
echo "Si es LoadBalancer, usá la IP externa; si no, podés hacer:"
echo "  kubectl -n ${NAMESPACE} port-forward svc/${RELEASE}-grafana 3000:80"

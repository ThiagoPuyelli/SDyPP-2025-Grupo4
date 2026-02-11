#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES_FILE="${SCRIPT_DIR}/values-loki-stack.yaml"
PROMTAIL_VALUES_FILE="${SCRIPT_DIR}/values-promtail.yaml"
PROMETHEUS_VALUES_FILE="${SCRIPT_DIR}/values-prometheus.yaml"
NAMESPACE="observability"
RELEASE="loki-stack"
PROMTAIL_RELEASE="promtail"
PROMETHEUS_RELEASE="prometheus"
ENV_FILE="${SCRIPT_DIR}/../vault/.env"
DASHBOARDS_DIR="${SCRIPT_DIR}/dashboards"

echo "Creando namespace ${NAMESPACE}..."
kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"

echo "Agregando repo de Helm grafana..."
helm repo add grafana https://grafana.github.io/helm-charts >/dev/null
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
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
LOKI_STS="$(kubectl -n "${NAMESPACE}" get sts -l app=loki,release="${RELEASE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -n "${LOKI_STS}" ]]; then
  kubectl -n "${NAMESPACE}" rollout status statefulset/"${LOKI_STS}" --timeout=300s || true
else
  echo "No se encontró StatefulSet de Loki (label app=loki,release=${RELEASE})."
fi

echo "Instalando/actualizando Prometheus como release separado (${PROMETHEUS_RELEASE})..."
helm upgrade --install "${PROMETHEUS_RELEASE}" prometheus-community/prometheus \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  -f "${PROMETHEUS_VALUES_FILE}"

kubectl -n "${NAMESPACE}" rollout status deploy/"${PROMETHEUS_RELEASE}"-server --timeout=300s || true

echo "Instalando/actualizando Promtail como release separado (${PROMTAIL_RELEASE})..."
helm upgrade --install "${PROMTAIL_RELEASE}" grafana/promtail \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  -f "${PROMTAIL_VALUES_FILE}"

PROMTAIL_DS="$(kubectl -n "${NAMESPACE}" get ds -l app.kubernetes.io/name=promtail,release="${PROMTAIL_RELEASE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
PROMTAIL_DEP="$(kubectl -n "${NAMESPACE}" get deploy -l app.kubernetes.io/name=promtail,release="${PROMTAIL_RELEASE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -n "${PROMTAIL_DS}" ]]; then
  kubectl -n "${NAMESPACE}" rollout status ds/"${PROMTAIL_DS}" --timeout=300s || true
elif [[ -n "${PROMTAIL_DEP}" ]]; then
  kubectl -n "${NAMESPACE}" rollout status deploy/"${PROMTAIL_DEP}" --timeout=300s || true
else
  echo "No se encontró Promtail (DaemonSet o Deployment con labels app=promtail,release=${PROMTAIL_RELEASE})."
fi

echo "Listo. Accedé a Grafana (usuario/pass tomados de ${ENV_FILE}) vía:"
echo "  kubectl -n ${NAMESPACE} get svc ${RELEASE}-grafana"
echo "Si es LoadBalancer, usá la IP externa; si no, podés hacer:"
echo "  kubectl -n ${NAMESPACE} port-forward svc/${RELEASE}-grafana 3000:80"

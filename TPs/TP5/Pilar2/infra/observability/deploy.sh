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

find_grafana_pod() {
  local pod=""
  pod="$(kubectl -n "${NAMESPACE}" get pods \
    -l app.kubernetes.io/name=grafana,app.kubernetes.io/instance="${RELEASE}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "${pod}" ]]; then
    echo "${pod}"
    return 0
  fi

  pod="$(kubectl -n "${NAMESPACE}" get pods \
    -l app=grafana,release="${RELEASE}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  echo "${pod}"
}

sync_grafana_admin_password() {
  local grafana_pod=""
  grafana_pod="$(find_grafana_pod)"

  if [[ -z "${grafana_pod}" ]]; then
    echo "No se encontró pod de Grafana para sincronizar contraseña admin." >&2
    return 1
  fi

  echo "Sincronizando contraseña admin en grafana.db (PVC persistente)..."
  if kubectl -n "${NAMESPACE}" exec "${grafana_pod}" -c grafana -- \
    grafana cli admin reset-admin-password "${GRAFANA_PASS}" >/dev/null 2>&1; then
    echo "Contraseña admin sincronizada correctamente."
    return 0
  fi

  if kubectl -n "${NAMESPACE}" exec "${grafana_pod}" -c grafana -- \
    grafana-cli admin reset-admin-password "${GRAFANA_PASS}" >/dev/null 2>&1; then
    echo "Contraseña admin sincronizada correctamente (fallback grafana-cli)."
    return 0
  fi

  echo "No se pudo sincronizar la contraseña admin en Grafana." >&2
  return 1
}

normalize_credential() {
  local value="${1-}"
  value="${value%$'\r'}"
  printf '%s' "${value}"
}

has_control_chars() {
  local value="${1-}"
  [[ "${value}" =~ [[:cntrl:]] ]]
}

has_edge_whitespace() {
  local value="${1-}"
  [[ "${value}" =~ ^[[:space:]] || "${value}" =~ [[:space:]]$ ]]
}

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

RAW_GRAFANA_USER="${GRAFANA_USER}"
RAW_GRAFANA_PASS="${GRAFANA_PASS}"
GRAFANA_USER="$(normalize_credential "${GRAFANA_USER}")"
GRAFANA_PASS="$(normalize_credential "${GRAFANA_PASS}")"

if [[ "${RAW_GRAFANA_USER}" != "${GRAFANA_USER}" || "${RAW_GRAFANA_PASS}" != "${GRAFANA_PASS}" ]]; then
  echo "Se removieron retornos de carro (\\r) de GRAFANA_USER/GRAFANA_PASS leídos desde ${ENV_FILE}."
fi

if has_control_chars "${GRAFANA_USER}" || has_control_chars "${GRAFANA_PASS}"; then
  echo "GRAFANA_USER/GRAFANA_PASS contienen caracteres de control no válidos en ${ENV_FILE}." >&2
  exit 1
fi

if has_edge_whitespace "${GRAFANA_USER}" || has_edge_whitespace "${GRAFANA_PASS}"; then
  echo "GRAFANA_USER/GRAFANA_PASS tienen espacios al inicio o al final en ${ENV_FILE}." >&2
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

echo "Reiniciando Grafana para aplicar cambios de credenciales en Secret..."
kubectl -n "${NAMESPACE}" rollout restart deploy/"${RELEASE}"-grafana || true

echo "Esperando a que Grafana y Loki estén listos..."
kubectl -n "${NAMESPACE}" rollout status deploy/"${RELEASE}"-grafana --timeout=300s
sync_grafana_admin_password
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

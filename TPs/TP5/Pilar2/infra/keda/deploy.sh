#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="keda"
RELEASE="keda"

echo "Agregando repo de Helm de KEDA..."
helm repo add kedacore https://kedacore.github.io/charts >/dev/null
helm repo update >/dev/null

echo "Instalando/actualizando KEDA..."

helm upgrade --install "${RELEASE}" kedacore/keda \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --set podIdentity.azureWorkload.enabled=false \
  --set podIdentity.aws.enabled=false

echo "Esperando a que KEDA esté listo..."

kubectl rollout status deployment/keda-operator -n "${NAMESPACE}" --timeout=300s
kubectl rollout status deployment/keda-operator-metrics-apiserver -n "${NAMESPACE}" --timeout=300s

echo "KEDA instalado correctamente 🚀"
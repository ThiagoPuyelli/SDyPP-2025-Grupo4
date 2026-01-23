#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Recorda tener el cliente de Linkerd instalado: https://linkerd.io/2.11/getting-started/#step-1-install-the-cli"


# 0) Instalar Gateway API CRDs (requerido por Linkerd)
if ! kubectl get crd gateways.gateway.networking.k8s.io >/dev/null 2>&1; then
  echo "Instalando Gateway API CRDs..."
  kubectl apply --server-side -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.4.0/standard-install.yaml
else
  echo "Gateway API CRDs ya instalados"
fi

# 1) Instalar Linkerd (si no está instalado)
if ! kubectl get ns linkerd >/dev/null 2>&1; then
  echo "Instalando Linkerd en el repo..."
  # 1.1) Instalar CRDs
  linkerd install --crds | kubectl apply -f -
  # 1.2) Instalar control plane
  linkerd install | kubectl apply -f -
  # 1.3) Check
  linkerd check --wait=300s
else
  echo "Linkerd ya instalado"
fi

# -------------------------
# 1) Crear namespaces
# -------------------------
for ns in blockchain pool; do
  kubectl create namespace $ns --dry-run=client -o yaml | kubectl apply -f -
done

# -------------------------
# 2) Labelar namespaces
# -------------------------
for ns in blockchain pool; do
  kubectl label namespace $ns linkerd.io/inject=enabled --overwrite
done
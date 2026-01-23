#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creando namespace argocd (si no existe)..."
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -

echo "Instalando Argo CD..."
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

echo "Esperando a que Argo CD esté listo..."
kubectl wait \
    --namespace argocd \
    --for=condition=Available \
    deployment/argocd-server \
    --timeout=300s

echo "Aplicando definición de Argo CD (Application / AppProject)..."
kubectl apply -n argocd -f "${SCRIPT_DIR}/argocd.yaml"

echo "Argo CD instalado y configurado"

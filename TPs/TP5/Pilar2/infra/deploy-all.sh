#!/usr/bin/env bash
set -euo pipefail

echo "Deploy completo del cluster..."

echo "|----- Deploying Vault..."
./vault/deploy.sh

echo "|----- Deploying Observability..."
./observability/deploy.sh

echo "|----- Deploying Argo CD..."
./argocd/deploy.sh

echo "|----- Deploy finalizado correctamente ✅"
echo "ArgoCD se encarga ahora del resto"
* pararse en /terraform y hacer:
```
terraform init
terraform plan
terraform apply
```

* levantar los nodos:
```
gcloud container clusters get-credentials sdeath-cluster --region us-east1-b --project sd-2025-459518

kubectl apply -f ../manifests/.
```
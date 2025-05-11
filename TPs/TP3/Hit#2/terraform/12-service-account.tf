# # 1. Crear la cuenta de servicio (ya lo tenés)
# resource "google_service_account" "kubernetes" {
#   account_id = "kubernetes"
# }

# # 2. Asignar roles necesarios para que pueda crear y administrar nodos
# resource "google_project_iam_member" "kubernetes_node_sa" {
#   project = var.project_id
#   role    = "roles/container.nodeServiceAccount"
#   member  = "serviceAccount:${google_service_account.kubernetes.email}"
# }

# resource "google_project_iam_member" "kubernetes_compute_admin" {
#   project = var.project_id
#   role    = "roles/compute.instanceAdmin.v1"
#   member  = "serviceAccount:${google_service_account.kubernetes.email}"
# }

# resource "google_project_iam_member" "kubernetes_sa_user" {
#   project = var.project_id
#   role    = "roles/iam.serviceAccountUser"
#   member  = "serviceAccount:${google_service_account.kubernetes.email}"
# }

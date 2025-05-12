# resource "google_storage_bucket_iam_member" "allow_kubernetes_sa_read" {
#   bucket = google_storage_bucket.bucket_imagenes.name
#   role   = "roles/storage.objectAdmin"
#   member = "serviceAccount:${google_service_account.kubernetes.email}"
# }

resource "google_service_account_iam_member" "ksa_to_gsa_binding" {
  service_account_id = google_service_account.kubernetes.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[default/bucket-reader]"
}


resource "google_storage_bucket_iam_member" "allow_kubernetes_sa_read" {
  bucket = google_storage_bucket.bucket_imagenes.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:kubernetes@${var.project_id}.iam.gserviceaccount.com"
}

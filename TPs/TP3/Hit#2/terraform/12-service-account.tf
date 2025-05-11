resource "google_storage_bucket_iam_member" "allow_kubernetes_sa_read" {
  bucket = google_storage_bucket.bucket_imagenes.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.kubernetes.email}"
}
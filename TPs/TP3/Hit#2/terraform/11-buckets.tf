resource "google_storage_bucket" "bucket_imagenes" {
  name     = "bucket_sobel"
  location = var.region
}
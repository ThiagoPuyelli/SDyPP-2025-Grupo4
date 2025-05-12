resource "google_storage_bucket" "bucket_imgs" {
  name     = "bucket_sobel2"
  location = var.region
}
# # https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_firewall
# resource "google_compute_firewall" "allow-ssh" {
#   name    = "allow-ssh"
#   network = google_compute_network.main.name

#   allow {
#     protocol = "tcp"
#     ports    = ["22"]
#   }

#   source_ranges = ["0.0.0.0/0"]
# }

# resource "google_compute_firewall" "allow_http_https" {
#   name    = "allow-http-https"
#   network = "default"

#   allow {
#     protocol = "tcp"
#     ports    = ["80", "443"]
#   }

#   direction     = "INGRESS"
#   source_ranges = ["0.0.0.0/0"]
#   target_tags   = ["http-server", "https-server"]
#   priority      = 1000
# }


# ---------------------
# ingress health checks
resource "google_compute_firewall" "allow_ingress_8000" {
  name    = "allow-gke-ingress-8000"
  network = google_compute_network.main.name

  direction = "INGRESS"

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16",
  ]

  target_service_accounts = [
    "kubernetes@spry-pier-480120-k1.iam.gserviceaccount.com"
  ]
}


resource "google_compute_firewall" "allow_gke_health_checks" {
  name    = "allow-gke-health-checks"
  network = google_compute_network.main.name

  direction = "INGRESS"

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "10256", "30000-32767"]
  }

  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]
}

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

resource "google_compute_address" "pool_queue_ip" {
    name   = "pool-queue-ip"
    region = var.region      # IP POR REGION, no global

    lifecycle {
        prevent_destroy = true
    }
}

resource "google_compute_address" "nginx_ip" {
    name   = "nginx-ip"
    region = var.region      # IP POR REGION, no global

    lifecycle {
        prevent_destroy = true
    }
}

output "nginx_ip" {
    value = google_compute_address.nginx_ip.address
}
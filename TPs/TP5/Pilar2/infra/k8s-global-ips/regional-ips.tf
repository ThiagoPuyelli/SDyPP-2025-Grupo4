resource "google_compute_address" "pool_queue_ip" {
    name   = "pool-queue-ip"
    region = var.region      # IP POR REGION, no global

    lifecycle {
        prevent_destroy = true
    }
}
resource "google_compute_instance" "worker_vm" {
  count        = 3
  name         = "worker-vm-${count.index}"
  machine_type = "e2-standard-2"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-2204-lts"
    }
  }

  network_interface {
    network    = google_compute_network.main.id
    subnetwork = google_compute_subnetwork.private.id
    access_config {}  # Esto les da IP pública. Podés quitarlo si sólo querés tráfico interno.
  }

  metadata_startup_script = replace(<<-EOT
  #!/bin/bash
  apt-get update
  apt-get install -y docker.io
  docker run -d --name consumidor \
    -e RABBITMQ_HOST=35.231.65.11 \
    -e REDIS_HOST=34.74.227.63 \
    -e JOINER_HOST=35.237.121.66 \
    docker.io/thiagopuyelli/consumidor_sobel:1.5
  EOT
  , "\r", "")


  service_account {
    email  = google_service_account.worker_vm_sa.email
    scopes = ["cloud-platform"]
  }

  tags = ["worker"]
}
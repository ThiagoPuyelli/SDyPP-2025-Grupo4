resource "google_container_node_pool" "mineros" {
  name       = "mineros"
  cluster    = google_container_cluster.primary.id
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 2
  }

  node_config {
    preemptible  = true
    machine_type = "e2-medium"
    disk_size_gb = 20

    labels = {
      role = "mineros"
    }

    service_account = google_service_account.kubernetes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

resource "google_container_node_pool" "blockchain" {
  name       = "blockchain"
  cluster    = google_container_cluster.primary.id
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 2
  }

  node_config {
    preemptible  = false
    machine_type = "e2-medium"
    disk_size_gb = 20

    labels = {
      role = "blockchain"
    }

    service_account = google_service_account.kubernetes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

resource "google_container_node_pool" "infraestructura" {
  name       = "infraestructura"
  cluster    = google_container_cluster.primary.id
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 2
  }

  node_config {
    machine_type = "e2-medium"
    preemptible  = false
    disk_size_gb = 20

    labels = {
      role = "infraestructura"
    }

    service_account = google_service_account.kubernetes.email
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}
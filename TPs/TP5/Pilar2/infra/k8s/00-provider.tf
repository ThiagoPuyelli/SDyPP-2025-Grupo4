
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">=5.26.0"
    }
    
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
  }
  # backend "gcs" {
  # }
  # required_version = ">= 1.7.5"
}

provider "google" {
  credentials = file(var.credentials_file_path)
  project     = var.project_id
  region      = var.region
  zone        = var.zone
}

data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(
    google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  )
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(
      google_container_cluster.primary.master_auth[0].cluster_ca_certificate
    )
  }
}
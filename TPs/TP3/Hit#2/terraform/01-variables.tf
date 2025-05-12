variable "region" {
  type    = string
  default = "us-east1"
}

variable "zone" {
  type    = string
  default = "us-east1-b"
}

variable "credentials_file_path" {
  description = "Path to GCP service account credentials file"
  default     = "../credentials/credentials.json"
}

variable "project_id" {
  type    = string
  default = "sd-2025-459518"
}

variable "cluster_name" {
  type        = string
  description = "Nombre del cluster de Kubernetes"
  default     = "sdeath-cluster"
}
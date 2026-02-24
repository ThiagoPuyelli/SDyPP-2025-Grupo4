variable "region" {
  type    = string
  default = "us-east1"
}

variable "zone" {
  type    = string
  default = "us-east1-b"
}

variable "credentials_file_path" {
  description = "Path al archivo de credenciales de la cuenta de servicio"
  default     = "../credentials/credentials.json"
}

variable "project_id" {
  type = string
}

variable "cluster_name" {
  type        = string
  description = "Nombre del cluster de Kubernetes"
}

variable "nginx_ip" {
  type        = string
  description = "IP estática del ingress nginx"
}

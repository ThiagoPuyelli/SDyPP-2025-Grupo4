variable "project_id" {
  description = "Id del proyecto"
  type        = string
}

variable "region" {
  description = "La región donde se desplegará el clúster"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Nombre del cluster de Kubernetes"
  type        = string
}

variable "initial_node_count" {
  description = "Cantidad de nodos iniciales"
  type        = number
  default     = 3
}
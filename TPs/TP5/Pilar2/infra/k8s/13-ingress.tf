resource "kubernetes_namespace" "ingress_nginx" {
    metadata {
        name = "ingress-nginx"
    }
}

variable "nginx_ip" {
    type        = string
    description = "IP estática del ingress nginx"
}

resource "helm_release" "nginx_ingress" {
    name       = "ingress-nginx"
    namespace  = kubernetes_namespace.ingress_nginx.metadata[0].name

    repository = "https://kubernetes.github.io/ingress-nginx"
    chart      = "ingress-nginx"

    depends_on = [
        kubernetes_namespace.ingress_nginx
    ]

    set {
        name  = "controller.service.type"
        value = "LoadBalancer"
    }

    set {
        name  = "controller.service.loadBalancerIP"
        value = var.nginx_ip
    }

    # Opcional pero recomendado en GKE
    set {
        name  = "controller.publishService.enabled"
        value = "true"
    }
}
resource "kubernetes_namespace" "ingress_nginx" {
    metadata {
        name = "ingress-nginx"
    }
}

data "terraform_remote_state" "ips" {
  backend = "gcs"
  config = {
    bucket = "terraform-state"
    prefix = "k8s-ips"
  }
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
        value = data.terraform_remote_state.ips.outputs.nginx_ip
    }

    # Opcional pero recomendado en GKE
    set {
        name  = "controller.publishService.enabled"
        value = "true"
    }
}
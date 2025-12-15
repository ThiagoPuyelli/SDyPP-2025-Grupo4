from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Contadores de negocio
tx_submitted_total = Counter(
    "coordinador_tx_submitted_total",
    "Transacciones recibidas por el coordinador",
    ["status"],
)

results_total = Counter(
    "coordinador_results_total",
    "Resultados de cadenas enviados por mineros",
    ["status"],
)

cycles_total = Counter(
    "coordinador_cycles_total",
    "Ciclos del coordinador",
    ["event"],
)

# Gauges de colas/estado
pending_queue_size = Gauge(
    "coordinador_pending_queue_size",
    "Tamaño de la cola de transacciones pendientes",
)

active_queue_size = Gauge(
    "coordinador_active_queue_size",
    "Tamaño de la cola de transacciones activas",
)

received_chains_size = Gauge(
    "coordinador_received_chains_size",
    "Cantidad de cadenas recibidas en el ciclo actual",
)


def record_tx(status: str) -> None:
    tx_submitted_total.labels(status=status).inc()


def record_result(status: str) -> None:
    results_total.labels(status=status).inc()


def record_cycle(event: str) -> None:
    cycles_total.labels(event=event).inc()


def update_queue_metrics(pending: int, active: int, received: int) -> None:
    pending_queue_size.set(pending)
    active_queue_size.set(active)
    received_chains_size.set(received)


def metrics_response():
    return generate_latest()

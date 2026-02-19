from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

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

miners_gauge = Gauge(
    "mining_workers_desired",
    "Desired number of mining workers"
)

mining_duration_seconds = Histogram(
    "coordinador_mining_duration_seconds",
    "Tiempo de minado por transaccion en la cadena ganadora del ciclo",
    ["prefix"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180, 300, float("inf")),
)

block_validation_duration_seconds = Histogram(
    "coordinador_block_validation_duration_seconds",
    "Tiempo de validacion de cada bloque recibido en el endpoint /results",
    ["prefix", "status"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, float("inf")),
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

def update_prefix_metric(prefix: str) -> None:
    difficulty = len(prefix)

    if difficulty <= 5:
        miners = 5
    elif difficulty == 6:
        miners = 4
    elif difficulty == 7:
        miners = 3
    elif difficulty == 8:
        miners = 2
    else:
        miners = 1

    miners_gauge.set(miners)


def record_mining_duration(prefix: str, seconds: float) -> None:
    if seconds < 0:
        return
    mining_duration_seconds.labels(prefix=prefix).observe(seconds)


def record_block_validation_duration(prefix: str, status: str, seconds: float) -> None:
    if seconds < 0:
        return
    block_validation_duration_seconds.labels(prefix=prefix, status=status).observe(seconds)


def metrics_response():
    return generate_latest()

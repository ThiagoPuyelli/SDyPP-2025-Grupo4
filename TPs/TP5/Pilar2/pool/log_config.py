import logging
import time
from datetime import timedelta

# Configuración básica global (fallback)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class MonotonicFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', hora_inicio=None, start_monotonic=None, desfase_monotonic=0):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.hora_inicio = hora_inicio
        self.start_monotonic = start_monotonic
        self.desfase_monotonic = desfase_monotonic

    def formatTime(self, record, datefmt=None):
        elapsed = time.monotonic() - self.start_monotonic - self.desfase_monotonic
        monotonic_now = self.hora_inicio + timedelta(seconds=elapsed)
        return monotonic_now.strftime(datefmt or self.default_time_format)

# Inicializamos con un logger "vacío" o nulo, para evitar NoneType
logger = logging.getLogger("minero")

def setup_logger_con_monotonic(hora_inicio, start_monotonic, desfase_monotonic=0):
    formatter = MonotonicFormatter(
        fmt='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        hora_inicio=hora_inicio,
        start_monotonic=start_monotonic,
        desfase_monotonic=desfase_monotonic
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    global logger  # para modificar la variable del módulo

    logger = logging.getLogger("minero")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    return logger

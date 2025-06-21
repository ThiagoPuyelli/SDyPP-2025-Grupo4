from datetime import timedelta
import time
from utils import get_last_interval_start


class MonotonicTime():
    def __init__(self, _hora_inicio):
        self.start_monotonic = time.monotonic()
        self.hora_inicio = _hora_inicio
        self.desfase_monotonic = (self.hora_inicio - get_last_interval_start(self.hora_inicio)).total_seconds()

    def get_hora_actual(self):
        ## RESTANDO ASI EL DESFASE PODEMOS FORZAR QUE ARRANQUE CON EL CICLO EN 0
        # elapsed = time.monotonic() - self.start_monotonic - self.desfase_monotonic
        elapsed = time.monotonic() - self.start_monotonic
        return self.hora_inicio + timedelta(seconds=elapsed)

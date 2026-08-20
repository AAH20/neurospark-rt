import random
from typing import List

class TemporalRateEncoder:
    """
    Asynchronous Temporal Rate Encoder.
    Converts continuous sensory signals / packet metrics into discrete microsecond spike trains.
    """
    def __init__(self, max_firing_rate_hz: float = 500.0):
        self.max_firing_rate_hz = max_firing_rate_hz

    def encode_continuous_to_current(self, signal_values: List[float], max_val: float = 100.0) -> List[float]:
        currents = []
        for val in signal_values:
            normalized = min(1.0, max(0.0, val / max(max_val, 1e-5)))
            # Higher signal = higher current injection into LIF neuron
            currents.append(normalized * 25.0) # 25mV current injection
        return currents

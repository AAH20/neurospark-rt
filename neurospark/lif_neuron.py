import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SpikeEvent:
    neuron_id: int
    timestamp_us: float # Microsecond timestamp
    membrane_voltage: float

class LIFNeuronLayer:
    """
    Vectorized Leaky Integrate-and-Fire (LIF) Spiking Neuron Layer.
    Simulates biological membrane dynamics:
    dV/dt = -(V - V_rest) / tau_mem + I_syn(t)
    When V >= V_thresh: emits SpikeEvent and resets to V_reset.
    """
    def __init__(
        self,
        num_neurons: int = 64,
        v_rest: float = -70.0,
        v_thresh: float = -55.0,
        v_reset: float = -75.0,
        tau_mem_ms: float = 20.0,
        refractory_period_us: float = 2000.0 # 2ms refractory
    ):
        self.num_neurons = num_neurons
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau_mem_ms = tau_mem_ms
        self.refractory_period_us = refractory_period_us

        # Neuron States
        self.voltages: List[float] = [v_rest] * num_neurons
        self.last_spike_times_us: List[float] = [-1e9] * num_neurons
        self.total_spikes_emitted = 0

    def step(self, current_inputs: List[float], dt_us: float = 100.0, current_time_us: float = 0.0) -> List[SpikeEvent]:
        dt_ms = dt_us / 1000.0
        decay = math.exp(-dt_ms / self.tau_mem_ms)
        spikes: List[SpikeEvent] = []

        for i in range(self.num_neurons):
            # Check refractory state
            if (current_time_us - self.last_spike_times_us[i]) < self.refractory_period_us:
                self.voltages[i] = self.v_reset
                continue

            # Leaky integration: V = V_rest + (V - V_rest)*decay + I_syn
            v = self.voltages[i]
            v_new = self.v_rest + (v - self.v_rest) * decay + current_inputs[i]

            if v_new >= self.v_thresh:
                # Emit Spike
                spikes.append(SpikeEvent(
                    neuron_id=i,
                    timestamp_us=current_time_us,
                    membrane_voltage=v_new
                ))
                self.voltages[i] = self.v_reset
                self.last_spike_times_us[i] = current_time_us
                self.total_spikes_emitted += 1
            else:
                self.voltages[i] = v_new

        return spikes

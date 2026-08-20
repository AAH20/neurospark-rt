import math
from typing import List, Dict, Tuple

class STDPSynapseMatrix:
    """
    Spike-Timing-Dependent Plasticity (STDP) On-Device Learning Matrix.
    Synaptic weight update rule:
    - If pre-spike occurs BEFORE post-spike (t_post - t_pre > 0): Long-Term Potentiation (LTP: W increases).
    - If pre-spike occurs AFTER post-spike (t_post - t_pre < 0): Long-Term Depression (LTD: W decreases).
    """
    def __init__(self, num_pre: int = 64, num_post: int = 64, a_plus: float = 0.05, a_minus: float = 0.055, tau_stdp_us: float = 20000.0):
        self.num_pre = num_pre
        self.num_post = num_post
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_stdp_us = tau_stdp_us

        # Initialize uniform synaptic weights [0.1 to 1.0]
        self.weights: List[List[float]] = [[0.5 for _ in range(num_post)] for _ in range(num_pre)]

    def apply_stdp_update(self, pre_neuron_id: int, post_neuron_id: int, t_pre_us: float, t_post_us: float):
        dt = t_post_us - t_pre_us
        
        if dt > 0:
            # Pre before Post: LTP
            dw = self.a_plus * math.exp(-dt / self.tau_stdp_us)
            self.weights[pre_neuron_id][post_neuron_id] = min(2.0, round(self.weights[pre_neuron_id][post_neuron_id] + dw, 4))
        elif dt < 0:
            # Post before Pre: LTD
            dw = self.a_minus * math.exp(dt / self.tau_stdp_us)
            self.weights[pre_neuron_id][post_neuron_id] = max(0.01, round(self.weights[pre_neuron_id][post_neuron_id] - dw, 4))

    def propagate_spikes(self, active_pre_spikes: List[int]) -> List[float]:
        """Calculates post-synaptic current injection: I_post = sum(W_ij for active pre-spikes)."""
        post_currents = [0.0] * self.num_post
        for pre_id in active_pre_spikes:
            for post_id in range(self.num_post):
                post_currents[post_id] += self.weights[pre_id][post_id] * 5.0 # Synaptic gain
        return post_currents

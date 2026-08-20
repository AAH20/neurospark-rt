"""
NeuroSpark-RT: Zero-Power Sub-Microsecond Spiking Neural Network (SNN) Runtime.
"""

from .lif_neuron import LIFNeuronLayer, SpikeEvent
from .stdp_synapse import STDPSynapseMatrix
from .event_encoder import TemporalRateEncoder
from .telemetry import NeuromorphicFinOpsTelemetry

__version__ = "0.1.0"
__all__ = [
    "LIFNeuronLayer",
    "SpikeEvent",
    "STDPSynapseMatrix",
    "TemporalRateEncoder",
    "NeuromorphicFinOpsTelemetry",
]

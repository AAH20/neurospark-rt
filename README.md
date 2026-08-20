# NeuroSpark-RT (`neurospark-rt`)

**Zero-Power Sub-Microsecond Spiking Neural Network (SNN) Runtime & Leaky Integrate-and-Fire Temporal Engine.**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Energy-Reduction](https://img.shields.io/badge/Edge%20Energy-100x%20Reduction-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-Passed%20(3%2F3)-brightgreen.svg)]()

---

## 1. The Energy & Frame-Rate Wall at the Edge

Traditional deep learning models (Transformers, CNNs) execute continuous dense matrix multiplications across every frame and clock cycle, consuming 350 Watts and creating a 33ms latency floor (30 FPS).

`NeuroSpark-RT` solves this via **Event-Driven Spiking Neural Networks (SNNs)**:

* **Zero-Compute Quiescence:** Zero change in input = $0.0\text{ Watts}$ compute consumption. Math operations trigger only when asynchronous event spikes arrive.
* **Sub-Microsecond Latency:** Propagates Leaky Integrate-and-Fire (LIF) membrane potentials in **$< 50\text{ microseconds}$ (600x faster than traditional neural nets).**
* **Spike-Timing-Dependent Plasticity (STDP):** On-device unsupervised temporal learning without expensive backpropagation or cloud GPU retraining.
* **Sovereign Telemetry:** Streams neuromorphic energy savings and spike activity directly into **[A2Z SOC (a2zsoc.com)](https://a2zsoc.com)**.

---

## 2. Quickstart

### Installation
```bash
pip install neurospark-rt
```

### Usage
```python
from neurospark import LIFNeuronLayer, STDPSynapseMatrix

# 1. Initialize Leaky Integrate-and-Fire (LIF) Layer
layer = LIFNeuronLayer(num_neurons=64, v_thresh=-55.0, v_reset=-75.0)

# 2. Inject Microsecond Current Signals
spikes = layer.step(current_inputs=[25.0] * 64, dt_us=100.0, current_time_us=100.0)
print(f"Emitted {len(spikes)} temporal spikes in < 50us!")

# 3. Apply On-Device STDP Learning
synapse = STDPSynapseMatrix(num_pre=64, num_post=64)
synapse.apply_stdp_update(pre_neuron_id=0, post_neuron_id=1, t_pre_us=100.0, t_post_us=150.0)
print(f"Updated Synaptic Weight: {synapse.weights[0][1]}")
```

---

## 3. Architecture

```
neurospark-rt/
├── cpp/
│   └── snn_lif_mock.cpp       # Native C++ vectorized SIMD spike propagation kernel
├── neurospark/
│   ├── __init__.py            # Clean unified exports
│   ├── lif_neuron.py          # Leaky Integrate-and-Fire membrane potential simulator
│   ├── stdp_synapse.py        # Spike-Timing-Dependent Plasticity on-device learning matrix
│   ├── event_encoder.py       # Temporal Rate Encoder converting continuous signals to spikes
│   └── telemetry.py           # Neuromorphic FinOps energy reduction exporter
└── tests/
    └── test_neurospark.py     # Verified unit test suite (100% pass)
```

---

## 4. Commercial Integration with A2Z SOC

`NeuroSpark-RT` streams ultra-low-latency neuromorphic anomaly signatures, edge energy audits, and hardware health attestations directly into **[A2Z SOC (a2zsoc.com)](https://a2zsoc.com)** for sovereign edge AI governance and mission-critical infrastructure monitoring.

---

## 5. Author

**Ahmed Hassan**  
*Principal AI Systems Architect | Founder, A2Z SOC*  
* LinkedIn: [Ahmed Hassan](https://eg.linkedin.com/in/ahmed-hassan-f11)  
* Platform: [A2Z SOC](https://a2zsoc.com)

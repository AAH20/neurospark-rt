import unittest
from neurospark.lif_neuron import LIFNeuronLayer, SpikeEvent
from neurospark.stdp_synapse import STDPSynapseMatrix
from neurospark.event_encoder import TemporalRateEncoder
from neurospark.telemetry import NeuromorphicFinOpsTelemetry

class TestNeuroSparkRT(unittest.TestCase):
    def setUp(self):
        self.layer = LIFNeuronLayer(num_neurons=8, v_thresh=-55.0, v_reset=-75.0)

    def test_lif_membrane_integration_and_spike_emission(self):
        # Inject strong sub-threshold current
        currents_no_spike = [2.0] * 8
        spikes1 = self.layer.step(currents_no_spike, dt_us=100.0, current_time_us=100.0)
        self.assertEqual(len(spikes1), 0)

        # Inject suprathreshold current -> Must emit spike event
        currents_spike = [25.0] * 8
        spikes2 = self.layer.step(currents_spike, dt_us=100.0, current_time_us=200.0)
        
        # Invariant: All 8 neurons fire SpikeEvents in sub-microsecond simulation
        self.assertEqual(len(spikes2), 8)
        self.assertEqual(spikes2[0].neuron_id, 0)
        self.assertEqual(self.layer.voltages[0], -75.0) # Reset voltage

    def test_stdp_synaptic_plasticity_ltp_and_ltd(self):
        synapse = STDPSynapseMatrix(num_pre=4, num_post=4, a_plus=0.1, a_minus=0.1)
        
        # 1. Pre before Post (t_pre = 100us, t_post = 200us) -> Long-Term Potentiation (LTP)
        initial_w = synapse.weights[0][0]
        synapse.apply_stdp_update(pre_neuron_id=0, post_neuron_id=0, t_pre_us=100.0, t_post_us=200.0)
        self.assertGreater(synapse.weights[0][0], initial_w)

        # 2. Post before Pre (t_pre = 300us, t_post = 200us) -> Long-Term Depression (LTD)
        synapse.apply_stdp_update(pre_neuron_id=1, post_neuron_id=1, t_pre_us=300.0, t_post_us=200.0)
        self.assertLess(synapse.weights[1][1], initial_w)

    def test_neuromorphic_finops_telemetry_audit(self):
        # Fire 5 spikes
        self.layer.step([30.0] * 8, dt_us=100.0, current_time_us=500.0)
        
        telemetry = NeuromorphicFinOpsTelemetry(self.layer)
        audit = telemetry.export_energy_audit(total_simulated_time_ms=100.0)
        
        self.assertGreater(audit["total_spikes_emitted"], 0)
        self.assertGreater(audit["energy_reduction_factor"], 10.0)
        self.assertIn("a2z_soc_edge_attestation", audit)

if __name__ == "__main__":
    unittest.main()

"""
Spike encoding methods for converting audio features to spike trains
"""
import numpy as np
import torch
from typing import Union, Tuple

class SpikeEncoder:
    """Base class for spike encoders"""
    def __init__(self, time_steps: int, dt: float = 1.0):
        self.time_steps = time_steps
        self.dt = dt
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RateEncoder(SpikeEncoder):
    """Rate-based spike encoding"""
    def __init__(self, time_steps: int, max_rate: float = 100.0, min_rate: float = 0.0):
        super().__init__(time_steps)
        self.max_rate = max_rate
        self.min_rate = min_rate

    def encode(self, data: np.ndarray) -> np.ndarray:
        # Ensure 2D: (features, time)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_features, n_time = data.shape

        # Normalize to [0, 1]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        rates = self.min_rate + (self.max_rate - self.min_rate) * data_norm

        # Initialize spikes array
        spikes = np.zeros((n_features, self.time_steps), dtype=np.float32)

        # Generate spikes per feature across time steps
        for t in range(self.time_steps):
            spikes[:, t] = np.random.rand(n_features) < (rates[:, t % n_time] * self.dt / 1000.0)

        return spikes


class LatencyEncoder(SpikeEncoder):
    """Latency-based spike encoding (time-to-first-spike)"""
    def __init__(self, time_steps: int, tau: float = 1.0):
        super().__init__(time_steps)
        self.tau = tau

    def encode(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_features, n_time = data.shape

        # Normalize to [0, 1]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        spike_times = self.time_steps * (1 - data_norm)
        spike_times = np.clip(spike_times, 0, self.time_steps - 1).astype(int)

        spikes = np.zeros((n_features, self.time_steps), dtype=np.float32)
        for i in range(n_features):
            spikes[i, spike_times[i, 0]] = 1.0
        return spikes


class DeltaEncoder(SpikeEncoder):
    """Delta modulation spike encoding - spikes on significant changes"""
    def __init__(self, time_steps: int, threshold: float = 0.1, refractory_period: int = 3):
        super().__init__(time_steps)
        self.threshold = threshold
        self.refractory_period = refractory_period

    def encode(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_features, n_time = data.shape

        # Resample if needed
        if n_time != self.time_steps:
            indices = np.linspace(0, n_time - 1, self.time_steps).astype(int)
            data = data[:, indices]

        spikes = np.zeros((n_features, self.time_steps), dtype=np.float32)
        last_spike_time = np.full(n_features, -self.refractory_period)
        accumulated_change = np.zeros(n_features)
        last_value = data[:, 0]

        for t in range(1, self.time_steps):
            delta = data[:, t] - last_value
            accumulated_change += delta
            for i in range(n_features):
                if t - last_spike_time[i] >= self.refractory_period:
                    if abs(accumulated_change[i]) >= self.threshold:
                        spikes[i, t] = 1.0
                        last_spike_time[i] = t
                        accumulated_change[i] = 0
                        last_value[i] = data[i, t]
        return spikes


class TemporalContrastEncoder(SpikeEncoder):
    """Temporal contrast encoding - emphasizes temporal changes"""
    def __init__(self, time_steps: int, alpha: float = 0.5):
        super().__init__(time_steps)
        self.alpha = alpha

    def encode(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_features, n_time = data.shape

        if n_time != self.time_steps:
            indices = np.linspace(0, n_time - 1, self.time_steps).astype(int)
            data = data[:, indices]

        filtered = np.zeros_like(data)
        filtered[:, 0] = data[:, 0]
        for t in range(1, self.time_steps):
            filtered[:, t] = self.alpha * filtered[:, t-1] + (1 - self.alpha) * data[:, t]

        on_channel = np.maximum(0, data - filtered)
        off_channel = np.maximum(0, filtered - data)
        combined = on_channel + off_channel

        rate_encoder = RateEncoder(self.time_steps)
        spikes = rate_encoder.encode(combined)
        return spikes


class PopulationEncoder(SpikeEncoder):
    """Population coding - multiple neurons per feature with different tuning curves"""
    def __init__(self, time_steps: int, n_neurons_per_feature: int = 10):
        super().__init__(time_steps)
        self.n_neurons = n_neurons_per_feature

    def encode(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_features, n_time = data.shape

        # Normalize
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        centers = np.linspace(0, 1, self.n_neurons)
        width = 1.0 / (self.n_neurons - 1)

        total_features = n_features * self.n_neurons
        spikes = np.zeros((total_features, self.time_steps), dtype=np.float32)

        for f in range(n_features):
            for n in range(self.n_neurons):
                response = np.exp(-0.5 * ((data_norm[f, :] - centers[n]) / width) ** 2)
                rate_encoder = RateEncoder(self.time_steps)
                neuron_spikes = rate_encoder.encode(response)
                spikes[f*self.n_neurons+n, :] = neuron_spikes[0, :]  # Only first feature row
        return spikes


class MultiModalEncoder:
    """Combine multiple encoding strategies"""
    def __init__(self, encoders: list, weights: list = None):
        self.encoders = encoders
        self.weights = weights or [1.0] * len(encoders)

    def encode(self, data: np.ndarray) -> np.ndarray:
        spike_trains = []
        for encoder, weight in zip(self.encoders, self.weights):
            spikes = encoder.encode(data) * weight
            spike_trains.append(spikes)
        combined = np.concatenate(spike_trains, axis=0)
        combined = (combined > 0).astype(np.float32)
        return combined


def plot_spike_train(spikes: np.ndarray, title: str = "Spike Train", save_path: str = None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    spike_times, neuron_ids = np.where(spikes)
    plt.scatter(spike_times, neuron_ids, s=1, c='black', marker='|')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron ID')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


# 🔍 Debug/Test Block
if __name__ == "__main__":
    data = np.random.rand(5, 50)
    encoder = RateEncoder(time_steps=50)
    spikes = encoder.encode(data)
    print("✅ Spike shape:", spikes.shape)
    plot_spike_train(spikes, "Rate Encoder Test")

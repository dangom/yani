from dataclasses import dataclass
from scipy.stats import gamma
import numpy as np


@dataclass
class DoubleGammaHRF:
    """Default values will return Glover's HRF for the auditory cortex."""

    peak_delay: float = 6.0
    peak_width: float = 0.9
    undershoot_delay: float = 12.0
    undershoot_width: float = 0.9
    positive_negative_ratio: float = 0.35

    def sample(self, t: np.array):
        peak = gamma.pdf(
            t, self.peak_delay / self.peak_width, loc=0, scale=self.peak_width
        )
        undershoot = gamma.pdf(
            t,
            self.undershoot_delay / self.undershoot_width,
            loc=0,
            scale=self.undershoot_width,
        )
        peak_norm = peak.max()
        undershoot_norm = undershoot.max()
        hrf = (
            peak / peak_norm
            - undershoot / undershoot_norm * self.positive_negative_ratio
        )
        return hrf

    def transform(self, signal: np.array, tr=0.02):
        sample_times = np.arange(0, 32, tr)  # Sample 30 seconds at 20ms intervals.
        convolved = np.convolve(signal, self.sample(sample_times))
        # The return size of the convolved signal is len(signal) + len(sample_times) +1.
        # To get what we want we need to discard the extra time.
        return convolved[: -(len(sample_times) - 1)]

@dataclass
class SinusoidalStimulus:
    start_offset: float = 14
    frequency: float = 0.1
    exponent: float = 1.
    luminance: float = 1.

    def sample(self, t: np.array):
        period = 1 / self.frequency
        _, time = np.divmod(self.start_offset, period)
        phase_offset = (time / period) * (2 * np.pi) + np.pi
        y_offset, y_norm = 1, 2
        osc = (
            (np.cos(2 * np.pi * self.frequency * t - phase_offset) + y_offset) / y_norm
        ) ** self.exponent * self.luminance
        osc = np.where(t < self.start_offset, 0, osc)
        return osc

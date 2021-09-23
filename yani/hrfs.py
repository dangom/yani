from dataclasses import dataclass
from scipy.stats import gamma
from scipy import signal
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

    def transform(self, timeseries: np.array, tr=0.02):
        sample_times = np.arange(0, 32, tr)  # Sample 30 seconds at 20ms intervals.
        convolved = np.convolve(timeseries, self.sample(sample_times))
        # The return size of the convolved signal is len(signal) + len(sample_times) +1.
        # To get what we want we need to discard the extra time.
        return convolved[: -(len(sample_times) - 1)]

@dataclass
class SinusoidalStimulus:
    start_offset: float = 0
    frequency: float = 0.2
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

def response_amplitude(signal: np.array, start_idx: int):
    "Get max amplitude from oscillation as peak - drought amplitude"
    cropped_signal = signal[start_idx:]
    return cropped_signal.max() - cropped_signal.min()

def response_delay(reference_stimulus: np.array, response: np.array, tr: float, frequency: float):
    """Get phase shift by computing the cross correlation between signals.
    Since we know that the stimulus comes first, we can constrain the estimation to show us a positive delay only, and return it constrained to the first cycle of the oscillatory response (since a delay of 2 cycles is meaningless).
    """

    correlation = signal.correlate(reference_stimulus, response, mode="same")[len(response)//2:]
    lags = signal.correlation_lags(reference_stimulus.size, response.size, mode="same")[len(response)//2:]
    lag = lags[np.argmax(correlation)]
    return 1/frequency  - (lag * tr)

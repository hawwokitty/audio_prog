import numpy as np
from scipy.signal import butter, lfilter

# Function to generate sine wave
def generate_sine_wave(frequency=440, amplitude=0.5, duration=1.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    return waveform, sample_rate

# Function to apply a low-pass filter (optional effect)
def apply_lowpass_filter(data, cutoff=1000, sample_rate=44100, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)
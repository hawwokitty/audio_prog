import numpy as np
from scipy.signal import butter, lfilter

# Function to generate sine wave
def generate_sine_wave(frequency, amplitude, duration=1.0, sample_rate=44100, 
                       attack=0.1, decay=0.1, sustain=0.7, release=0.2):
    """ Generate a sine wave with a flexible ADSR envelope where duration adjusts dynamically """

    # Calculate total duration based on ADSR phases
    total_duration = attack + decay + sustain + release
    t = np.linspace(0, total_duration, int(sample_rate * total_duration), endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)

    # Compute sample counts
    attack_samples = int(sample_rate * attack)
    decay_samples = int(sample_rate * decay)
    sustain_samples = int(sample_rate * sustain)
    release_samples = int(sample_rate * release)

    env = np.ones(len(waveform))

    # Apply ADSR envelope
    env[:attack_samples] = np.linspace(0, 1, attack_samples)  # Attack
    env[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples)  # Decay
    env[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = sustain  # Sustain
    env[-release_samples:] = np.linspace(sustain, 0, release_samples)  # Release

    return waveform * env, sample_rate, total_duration

# Function to apply a low-pass filter (optional effect)
def apply_lowpass_filter(data, cutoff=1000, sample_rate=44100, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)
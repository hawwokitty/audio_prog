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

def generate_square_wave(frequency, amplitude, duty_cycle=0.5, duration=1.0, sample_rate=44100):
    """Generate a square wave with a specified duty cycle."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = amplitude * np.sign(np.sin(2 * np.pi * frequency * t) + (2 * duty_cycle - 1))
    return waveform, sample_rate, duration  # Return all three values


def generate_noise(amplitude, duration=1.0, sample_rate=44100):
    """Generate white noise."""
    return amplitude * np.random.uniform(-1, 1, int(sample_rate * duration))

# Function to apply a low-pass filter (optional effect)
def apply_lowpass_filter(data, cutoff=1000, sample_rate=44100, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def convert_to_bit_depth(samples, bit_depth):
    """Convert samples to the specified bit depth."""
    if bit_depth == 8:
        # 8-bit range = [0, 255]
        return np.uint8((samples + 1) * 127.5)
    elif bit_depth == 16:
        # 16-bit range = [-32768, 32767]
        return np.int16(samples * 32767)
    else:
        raise ValueError("Unsupported bit depth. Use 8 or 16.")
    
def generate_sawtooth_wave(frequency, amplitude, duration=1.0, sample_rate=44100):
    """Generate a sawtooth wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 2 * (t * frequency - np.floor(t * frequency)) - 1  # Sawtooth formula
    return amplitude * waveform, sample_rate, duration  # Return all three values

import numpy as np

def generate_vibrato(frequency, amplitude, vibrato_rate=5.0, vibrato_depth=0.02, duration=1.0, sample_rate=44100):
    """Generate a sine wave with vibrato effect."""

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a low-frequency oscillation (LFO) for vibrato effect
    vibrato = vibrato_depth * frequency * np.sin(2 * np.pi * vibrato_rate * t)

    # Apply vibrato modulation to the main sine wave frequency
    waveform = amplitude * np.sin(2 * np.pi * (frequency + vibrato) * t)

    return waveform, sample_rate, duration

def apply_distortion(waveform, gain=5.0, mix=0.5):
    """Apply a soft-clipping distortion effect to an audio waveform."""
    # Apply gain
    distorted = waveform * gain
    
    # Soft clipping (tanh-based)
    distorted = np.tanh(distorted)

    # Blend distorted signal with original
    output = (1 - mix) * waveform + mix * distorted
    return output
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QPushButton, QLabel, QCheckBox
from PyQt6.QtCore import Qt
from audio_utils import generate_sine_wave, apply_lowpass_filter, generate_square_wave, generate_sawtooth_wave, generate_noise, convert_to_bit_depth

# GUI Class
class AudioApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Synthesizer")
        self.setGeometry(100, 100, 500, 400)
        # ADSR Default Values
        self.attack = 0.1
        self.decay = 0.1
        self.sustain = 0.7
        self.release = 0.2


        # Default values
        self.frequency = 440
        self.amplitude = 0.5
        self.cutoff = 1000  # Default low-pass cutoff
        self.duration = 1.0  # Default duration in seconds

        # Central Widget & Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Frequency Slider
        self.freq_label = QLabel(f"Frequency: {self.frequency} Hz")
        self.layout.addWidget(self.freq_label)
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setMinimum(100)
        self.freq_slider.setMaximum(2000)
        self.freq_slider.setValue(self.frequency)
        self.freq_slider.valueChanged.connect(self.update_frequency)
        self.layout.addWidget(self.freq_slider)

        # Amplitude Slider
        self.amp_label = QLabel(f"Amplitude: {self.amplitude}")
        self.layout.addWidget(self.amp_label)
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setMinimum(1)   # 0.1 * 10
        self.amp_slider.setMaximum(10)  # 1.0 * 10
        self.amp_slider.setValue(int(self.amplitude * 10))
        self.amp_slider.valueChanged.connect(self.update_amplitude)
        self.layout.addWidget(self.amp_slider)

        # Low-Pass Filter Slider
        self.cutoff_label = QLabel(f"Low-pass Cutoff: {self.cutoff} Hz")
        self.layout.addWidget(self.cutoff_label)
        self.cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self.cutoff_slider.setMinimum(100)  # Min 100 Hz
        self.cutoff_slider.setMaximum(5000)  # Max 5000 Hz
        self.cutoff_slider.setValue(self.cutoff)
        self.cutoff_slider.setTickInterval(100)
        self.cutoff_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.cutoff_slider.valueChanged.connect(self.update_cutoff)
        self.layout.addWidget(self.cutoff_slider)
        
        # Attack Slider
        self.attack_label = QLabel(f"Attack: {self.attack} s")
        self.layout.addWidget(self.attack_label)
        self.attack_slider = QSlider(Qt.Orientation.Horizontal)
        self.attack_slider.setMinimum(1)  # 0.1s * 10
        self.attack_slider.setMaximum(50)  # 5.0s * 10
        self.attack_slider.setValue(int(self.attack * 10))
        self.attack_slider.valueChanged.connect(self.update_attack)
        self.layout.addWidget(self.attack_slider)

        # Decay Slider
        self.decay_label = QLabel(f"Decay: {self.decay} s")
        self.layout.addWidget(self.decay_label)
        self.decay_slider = QSlider(Qt.Orientation.Horizontal)
        self.decay_slider.setMinimum(1)
        self.decay_slider.setMaximum(50)
        self.decay_slider.setValue(int(self.decay * 10))
        self.decay_slider.valueChanged.connect(self.update_decay)
        self.layout.addWidget(self.decay_slider)

        # Sustain Slider
        self.sustain_label = QLabel(f"Sustain: {self.sustain}")
        self.layout.addWidget(self.sustain_label)
        self.sustain_slider = QSlider(Qt.Orientation.Horizontal)
        self.sustain_slider.setMinimum(0)  # 0.0 to 1.0 (scaled by 10)
        self.sustain_slider.setMaximum(10)
        self.sustain_slider.setValue(int(self.sustain * 10))
        self.sustain_slider.valueChanged.connect(self.update_sustain)
        self.layout.addWidget(self.sustain_slider)

        # Release Slider
        self.release_label = QLabel(f"Release: {self.release} s")
        self.layout.addWidget(self.release_label)
        self.release_slider = QSlider(Qt.Orientation.Horizontal)
        self.release_slider.setMinimum(1)
        self.release_slider.setMaximum(50)
        self.release_slider.setValue(int(self.release * 10))
        self.release_slider.valueChanged.connect(self.update_release)
        self.layout.addWidget(self.release_slider)

        # Checkbox for 8-bit conversion
        self.bit_depth_checkbox = QCheckBox("Convert to 8-bit")
        self.layout.addWidget(self.bit_depth_checkbox)
        
        # Checkbox for lowpass filter
        self.lowpass_checkbox = QCheckBox("Apply lowpass filter")
        self.layout.addWidget(self.lowpass_checkbox)

        # Play square Button
        self.play_square_button = QPushButton("Play Square Sound")
        self.play_square_button.clicked.connect(self.play_square_sound)
        self.layout.addWidget(self.play_square_button)
        
        # Play sine Button
        self.play_sine_button = QPushButton("Play Sine Sound")
        self.play_sine_button.clicked.connect(self.play_sine_sound)
        self.layout.addWidget(self.play_sine_button)
        
        # Play sawtooth Button
        self.play_sawtooth_button = QPushButton("Play Saw Tooth Sound")
        self.play_sawtooth_button.clicked.connect(self.play_sawtooth_sound)
        self.layout.addWidget(self.play_sawtooth_button)

        # Plot Button
        self.plot_button = QPushButton("Show Waveform")
        self.plot_button.clicked.connect(self.plot_waveform)
        self.layout.addWidget(self.plot_button)

    # Update Frequency
    def update_frequency(self, value):
        self.frequency = value
        self.freq_label.setText(f"Frequency: {self.frequency} Hz")

    # Update Amplitude
    def update_amplitude(self, value):
        self.amplitude = value / 10.0
        self.amp_label.setText(f"Amplitude: {self.amplitude}")

    # Update Low-Pass Cutoff Frequency
    def update_cutoff(self, value):
        self.cutoff = value
        self.cutoff_label.setText(f"Low-pass Cutoff: {self.cutoff} Hz")
        
    def update_attack(self, value):
        self.attack = value / 10.0
        self.attack_label.setText(f"Attack: {self.attack} s")

    def update_decay(self, value):
        self.decay = value / 10.0
        self.decay_label.setText(f"Decay: {self.decay} s")

    def update_sustain(self, value):
        self.sustain = value / 10.0
        self.sustain_label.setText(f"Sustain: {self.sustain}")

    def update_release(self, value):
        self.release = value / 10.0
        self.release_label.setText(f"Release: {self.release} s")
        

    # Play the generated square sound 
    def play_square_sound(self):
        waveform, sr, total_duration = generate_square_wave(
            self.frequency, 
            self.amplitude
        )
        
        edited_waveform = waveform
    
        # Apply low-pass filter
        if self.lowpass_checkbox.isChecked():
            waveform = apply_lowpass_filter(waveform, cutoff=self.cutoff, sample_rate=sr)
        
        # If the checkbox is checked, convert the waveform to 8-bit
        if self.bit_depth_checkbox.isChecked():
            edited_waveform = convert_to_bit_depth(waveform, bit_depth=8)
            # Properly convert from uint8 (0 to 255) to int16 (-32768 to 32767)
            edited_waveform = (edited_waveform.astype(np.int16) - 128) * 256

        # Play the generated sound
        sd.play(edited_waveform, sr)

        # Save the sound file with correct duration
        sf.write("generated_audio.wav", edited_waveform, sr)
        print("Waveform Stats:")
        print("Min:", np.min(waveform), "Max:", np.max(waveform), "Mean:", np.mean(waveform))
        print("Sample Rate:", sr, "Duration:", total_duration)
        
    # Play the generated saw tooth sound 
    def play_sawtooth_sound(self):
        waveform, sr, total_duration = generate_sawtooth_wave(
            self.frequency, 
            self.amplitude
        )
        
        edited_waveform = waveform
    
        # Apply low-pass filter
        if self.lowpass_checkbox.isChecked():
            waveform = apply_lowpass_filter(waveform, cutoff=self.cutoff, sample_rate=sr)
        
        # If the checkbox is checked, convert the waveform to 8-bit
        if self.bit_depth_checkbox.isChecked():
            edited_waveform = convert_to_bit_depth(waveform, bit_depth=8)
            # Properly convert from uint8 (0 to 255) to int16 (-32768 to 32767)
            edited_waveform = (edited_waveform.astype(np.int16) - 128) * 256

        # Play the generated sound
        sd.play(edited_waveform, sr)

        # Save the sound file with correct duration
        sf.write("generated_audio.wav", edited_waveform, sr)
        print("Waveform Stats:")
        print("Min:", np.min(waveform), "Max:", np.max(waveform), "Mean:", np.mean(waveform))
        print("Sample Rate:", sr, "Duration:", total_duration)
        
    # Play the generated sine sound
    def play_sine_sound(self):
        waveform, sr, total_duration = generate_sine_wave(
            self.frequency, 
            self.amplitude, 
            attack=self.attack, 
            decay=self.decay, 
            sustain=self.sustain, 
            release=self.release
        )
        
        edited_waveform = waveform
    
        # Apply low-pass filter
        if self.lowpass_checkbox.isChecked():
            edited_waveform = apply_lowpass_filter(waveform, cutoff=self.cutoff, sample_rate=sr)
        
        # If the checkbox is checked, convert the waveform to 8-bit
        if self.bit_depth_checkbox.isChecked():
            edited_waveform = convert_to_bit_depth(waveform, bit_depth=8)
            # Properly convert from uint8 (0 to 255) to int16 (-32768 to 32767)
            edited_waveform = (edited_waveform.astype(np.int16) - 128) * 256

        # Play the generated sound
        sd.play(edited_waveform, sr)

        # Save the sound file with correct duration
        sf.write("generated_audio.wav", edited_waveform, sr)
        print("Waveform Stats:")
        print("Min:", np.min(waveform), "Max:", np.max(waveform), "Mean:", np.mean(waveform))
        print("Sample Rate:", sr, "Duration:", total_duration)


    
    def plot_waveform(self):
        waveform, sr, total_duration = generate_sine_wave(
            self.frequency, 
            self.amplitude, 
            attack=self.attack, 
            decay=self.decay, 
            sustain=self.sustain, 
            release=self.release
        )
        
        edited_waveform = waveform

       # Apply low-pass filter
        if self.lowpass_checkbox.isChecked():
            edited_waveform = apply_lowpass_filter(waveform, cutoff=self.cutoff, sample_rate=sr)
        
        # If the checkbox is checked, convert the waveform to 8-bit
        if self.bit_depth_checkbox.isChecked():
            edited_waveform = convert_to_bit_depth(waveform, bit_depth=8)
            # Properly convert from uint8 (0 to 255) to int16 (-32768 to 32767)
            edited_waveform = (edited_waveform.astype(np.int16) - 128) * 256

        # Create time axis in seconds
        time_axis = np.linspace(0, total_duration, len(edited_waveform))

        plt.figure(figsize=(8, 4))
        plt.plot(time_axis, edited_waveform, label="Filtered Waveform")
        plt.title(f"Waveform ({self.frequency} Hz) with ADSR Envelope")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.show()


# Run Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioApp()
    window.show()
    sys.exit(app.exec())

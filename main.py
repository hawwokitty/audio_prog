import sys
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QPushButton, QLabel
from PyQt6.QtCore import Qt
from audio_utils import generate_sine_wave, apply_lowpass_filter

# GUI Class
class AudioApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Synthesizer")
        self.setGeometry(100, 100, 500, 400)

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

        # Play Button
        self.play_button = QPushButton("Play Sound")
        self.play_button.clicked.connect(self.play_sound)
        self.layout.addWidget(self.play_button)

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

    # Play the generated sound
    def play_sound(self):
        waveform, sr = generate_sine_wave(self.frequency, self.amplitude, duration=self.duration)

        # Apply low-pass filter with slider value
        filtered_waveform = apply_lowpass_filter(waveform, cutoff=self.cutoff, sample_rate=sr)

        sd.play(filtered_waveform, sr)
        
        # Save to file
        sf.write("generated_audio.wav", filtered_waveform, sr)
    
    def plot_waveform(self):
        waveform, sr = generate_sine_wave(self.frequency, self.amplitude, duration=self.duration)
        
        # Apply low-pass filter
        filtered_waveform = apply_lowpass_filter(waveform, cutoff=self.cutoff, sample_rate=sr)

        plt.figure(figsize=(6, 3))
        plt.plot(filtered_waveform[:1000])  # Show first 1000 samples
        plt.title(f"Filtered Sine Wave ({self.frequency} Hz, Low-pass at {self.cutoff} Hz)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

# Run Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioApp()
    window.show()
    sys.exit(app.exec())

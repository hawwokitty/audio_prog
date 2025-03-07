import sys
import numpy as np
import sounddevice as sd
import soundfile as sf

import os
os.environ["QT_API"] = "PyQt6"  # Force Matplotlib to use PyQt6

import matplotlib
matplotlib.use("QtAgg")  # Ensure it's using the correct Qt backend


import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QPushButton, QLabel, QCheckBox, QSizePolicy
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QFont, QFontDatabase
from audio_utils import generate_sine_wave, apply_lowpass_filter, generate_square_wave, generate_sawtooth_wave, generate_noise, apply_distortion, generate_vibrato, convert_to_bit_depth

# Matplotlib Canvas for embedding in PyQt
class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(WaveformCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Make the canvas expandable
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def plot_waveform(self, waveform, sr, duration, title="Waveform"):
        """Update the plot with new waveform data"""
        self.axes.clear()
        time_axis = np.linspace(0, duration, len(waveform))
        self.axes.plot(time_axis, waveform, color="#ff69b4", linewidth=2)  # Pink lines
        self.axes.set_facecolor("#fff0f5")  # Light pink background
        
        # Set pink title and labels
        self.axes.set_title(title, color="#ff69b4", fontsize=14)  
        self.axes.set_xlabel("Time (seconds)", color="#ff69b4", fontsize=12)
        self.axes.set_ylabel("Amplitude", color="#ff69b4", fontsize=12)

        # Set tick colors to pink
        self.axes.tick_params(axis="x", colors="#ff69b4")
        self.axes.tick_params(axis="y", colors="#ff69b4")

        self.axes.grid(True)
        self.fig.tight_layout()
        self.draw()


# GUI Class
class AudioApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🌸 Cute Synth 🌸")
        self.setWindowIcon(QIcon("cute_icon.png"))
        self.setGeometry(100, 100, 800, 600)  # Increased size to accommodate waveform
        
        # font_id = QFontDatabase.addApplicationFont("SourGummy-VariableFont_wdth,wght.ttf")
        # if font_id == -1:
        #     print("Failed to load font!")

        # font_family = QFontDatabase.applicationFontFamilies(font_id)[0]  # Get the font name
        # # Apply the font
        # cute_font = QFont(font_family, 16)
        
        self.setStyleSheet("""
    QWidget {
        background-color: #ffe4e1;  /* Light pink */
        font-family: "Trebuchet MS";
    }
    QLabel {
        color: #ff69b4;  /* Hot pink */
        font-size: 16px;
    }
    QPushButton {
        background-color: #ffb6c1;  /* Soft pink */
        border-radius: 10px;
        padding: 5px;
    }
    QPushButton:hover {
        background-color: #ff69b4;
        color: white;
    }
    QSlider::groove:horizontal {
        background: #ffb6c1;
        height: 10px;
    }
    QSlider::handle:horizontal {
        background: #ff69b4;
        width: 20px;
    }
    QCheckBox {
        color: #ff69b4;
    }
    QCheckBox::indicator {
        width: 16px;  /* Size of the box */
        height: 16px;
        border-radius: 4px;  /* Slightly rounded corners */
        border: 2px solid #ff69b4;  /* Hot pink border */
        background-color: #ffb6c1;  /* Soft pink */
    }
    QCheckBox::indicator:hover {
        background-color: #ff69b4;  /* Hot pink when hovered */
    }
    
    QCheckBox::indicator:checked {
        background-color: #ff1493;  /* Deep pink when checked */
        border: 2px solid #ff69b4;  /* Keep border hot pink */
    }
""")

        
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
        self.current_waveform_type = "sine"  # Track current waveform type

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout()  # Use horizontal layout for controls + waveform
        self.main_widget.setLayout(self.main_layout)
        
        # Left panel for controls
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout()
        self.controls_widget.setLayout(self.controls_layout)
        
        # Right panel for waveform
        self.waveform_canvas = WaveformCanvas(self.main_widget)
        
        # Add both panels to main layout
        self.main_layout.addWidget(self.controls_widget, 1)  # Controls take 1 part
        self.main_layout.addWidget(self.waveform_canvas, 2)  # Waveform takes 2 parts

        # ===== CONTROLS =====
        # Frequency Slider
        self.freq_label = QLabel(f"Frequency: {self.frequency} Hz")
        # self.freq_label.setFont(cute_font)
        self.controls_layout.addWidget(self.freq_label)
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setMinimum(100)
        self.freq_slider.setMaximum(2000)
        self.freq_slider.setValue(self.frequency)
        self.freq_slider.valueChanged.connect(self.update_frequency)
        self.controls_layout.addWidget(self.freq_slider)

        # Amplitude Slider
        self.amp_label = QLabel(f"Amplitude: {self.amplitude}")
        self.controls_layout.addWidget(self.amp_label)
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setMinimum(1)   # 0.1 * 10
        self.amp_slider.setMaximum(10)  # 1.0 * 10
        self.amp_slider.setValue(int(self.amplitude * 10))
        self.amp_slider.valueChanged.connect(self.update_amplitude)
        self.controls_layout.addWidget(self.amp_slider)

        # Low-Pass Filter Slider
        self.cutoff_label = QLabel(f"Low-pass Cutoff: {self.cutoff} Hz")
        self.controls_layout.addWidget(self.cutoff_label)
        self.cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self.cutoff_slider.setMinimum(100)  # Min 100 Hz
        self.cutoff_slider.setMaximum(5000)  # Max 5000 Hz
        self.cutoff_slider.setValue(self.cutoff)
        self.cutoff_slider.setTickInterval(100)
        self.cutoff_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.cutoff_slider.valueChanged.connect(self.update_cutoff)
        self.controls_layout.addWidget(self.cutoff_slider)
        
        # Attack Slider
        self.attack_label = QLabel(f"Attack: {self.attack} s")
        self.controls_layout.addWidget(self.attack_label)
        self.attack_slider = QSlider(Qt.Orientation.Horizontal)
        self.attack_slider.setMinimum(1)  # 0.1s * 10
        self.attack_slider.setMaximum(50)  # 5.0s * 10
        self.attack_slider.setValue(int(self.attack * 10))
        self.attack_slider.valueChanged.connect(self.update_attack)
        self.controls_layout.addWidget(self.attack_slider)

        # Decay Slider
        self.decay_label = QLabel(f"Decay: {self.decay} s")
        self.controls_layout.addWidget(self.decay_label)
        self.decay_slider = QSlider(Qt.Orientation.Horizontal)
        self.decay_slider.setMinimum(1)
        self.decay_slider.setMaximum(50)
        self.decay_slider.setValue(int(self.decay * 10))
        self.decay_slider.valueChanged.connect(self.update_decay)
        self.controls_layout.addWidget(self.decay_slider)

        # Sustain Slider
        self.sustain_label = QLabel(f"Sustain: {self.sustain}")
        self.controls_layout.addWidget(self.sustain_label)
        self.sustain_slider = QSlider(Qt.Orientation.Horizontal)
        self.sustain_slider.setMinimum(0)  # 0.0 to 1.0 (scaled by 10)
        self.sustain_slider.setMaximum(10)
        self.sustain_slider.setValue(int(self.sustain * 10))
        self.sustain_slider.valueChanged.connect(self.update_sustain)
        self.controls_layout.addWidget(self.sustain_slider)

        # Release Slider
        self.release_label = QLabel(f"Release: {self.release} s")
        self.controls_layout.addWidget(self.release_label)
        self.release_slider = QSlider(Qt.Orientation.Horizontal)
        self.release_slider.setMinimum(1)
        self.release_slider.setMaximum(50)
        self.release_slider.setValue(int(self.release * 10))
        self.release_slider.valueChanged.connect(self.update_release)
        self.controls_layout.addWidget(self.release_slider)

        # Checkbox for 8-bit conversion
        self.bit_depth_checkbox = QCheckBox("Convert to 8-bit")
        self.bit_depth_checkbox.stateChanged.connect(self.update_waveform)
        self.controls_layout.addWidget(self.bit_depth_checkbox)
        
        # Checkbox for lowpass filter
        self.lowpass_checkbox = QCheckBox("Apply lowpass filter")
        self.lowpass_checkbox.stateChanged.connect(self.update_waveform)
        self.controls_layout.addWidget(self.lowpass_checkbox)
        
        # Checkbox for noise filter
        self.noise_checkbox = QCheckBox("Apply noise filter")
        self.noise_checkbox.stateChanged.connect(self.update_waveform)
        self.controls_layout.addWidget(self.noise_checkbox)
        
        # Checkbox for vibrato filter
        self.vibrato_checkbox = QCheckBox("Apply vibrato filter")
        self.vibrato_checkbox.stateChanged.connect(self.update_waveform)
        self.controls_layout.addWidget(self.vibrato_checkbox)
        
        # Checkbox for distortion
        self.distortion_checkbox = QCheckBox("Apply distortion filter")
        self.distortion_checkbox.stateChanged.connect(self.update_waveform)
        self.controls_layout.addWidget(self.distortion_checkbox)

        # Play square Button
        self.play_square_button = QPushButton("Play Square Sound")
        self.play_square_button.clicked.connect(self.play_square_sound)
        self.controls_layout.addWidget(self.play_square_button)
        
        # Play sine Button
        self.play_sine_button = QPushButton("Play Sine Sound")
        self.play_sine_button.clicked.connect(self.play_sine_sound)
        self.controls_layout.addWidget(self.play_sine_button)
        
        # Play sawtooth Button
        self.play_sawtooth_button = QPushButton("Play Saw Tooth Sound")
        self.play_sawtooth_button.clicked.connect(self.play_sawtooth_sound)
        self.controls_layout.addWidget(self.play_sawtooth_button)
        
        # Initial waveform display
        self.update_waveform()

    # ===== UPDATE FUNCTIONS =====
    def update_frequency(self, value):
        self.frequency = value
        self.freq_label.setText(f"Frequency: {self.frequency} Hz")
        self.update_waveform()

    def update_amplitude(self, value):
        self.amplitude = value / 10.0
        self.amp_label.setText(f"Amplitude: {self.amplitude}")
        self.update_waveform()

    def update_cutoff(self, value):
        self.cutoff = value
        self.cutoff_label.setText(f"Low-pass Cutoff: {self.cutoff} Hz")
        self.update_waveform()
        
    def update_attack(self, value):
        self.attack = value / 10.0
        self.attack_label.setText(f"Attack: {self.attack} s")
        self.update_waveform()

    def update_decay(self, value):
        self.decay = value / 10.0
        self.decay_label.setText(f"Decay: {self.decay} s")
        self.update_waveform()

    def update_sustain(self, value):
        self.sustain = value / 10.0
        self.sustain_label.setText(f"Sustain: {self.sustain}")
        self.update_waveform()

    def update_release(self, value):
        self.release = value / 10.0
        self.release_label.setText(f"Release: {self.release} s")
        self.update_waveform()
    
    # ===== MAIN WAVEFORM UPDATE FUNCTION =====
    def update_waveform(self):
        # Generate waveform based on current type
        if self.current_waveform_type == "sine":
            waveform, sr, total_duration = generate_sine_wave(
                self.frequency, 
                self.amplitude, 
                attack=self.attack, 
                decay=self.decay, 
                sustain=self.sustain, 
                release=self.release
            )
        elif self.current_waveform_type == "square":
            waveform, sr, total_duration = generate_square_wave(
                self.frequency, 
                self.amplitude
            )
        elif self.current_waveform_type == "sawtooth":
            waveform, sr, total_duration = generate_sawtooth_wave(
                self.frequency, 
                self.amplitude
            )
        
        edited_waveform = waveform.copy()
        
        # Apply low-pass filter if checked
        if self.lowpass_checkbox.isChecked():
            edited_waveform = apply_lowpass_filter(edited_waveform, cutoff=self.cutoff, sample_rate=sr)
        
        # Apply distortion filter if checked
        if self.distortion_checkbox.isChecked():
            edited_waveform = apply_distortion(edited_waveform)
            
        # Apply noise filter if checked
        if self.noise_checkbox.isChecked():
            edited_waveform = generate_noise(amplitude=self.amplitude)
        
        # Apply vibrato filter if checked
        if self.vibrato_checkbox.isChecked():
            edited_waveform, sr, total_duration = generate_vibrato(frequency=self.frequency, amplitude=self.amplitude)
        
        # Apply bit depth conversion if checked
        if self.bit_depth_checkbox.isChecked():
            edited_waveform = convert_to_bit_depth(edited_waveform, bit_depth=8)
            # Convert from uint8 to appropriate range
            edited_waveform = (edited_waveform.astype(np.int16) - 128) * 256
        
        # Update the waveform display
        title = f"{self.current_waveform_type.capitalize()} Wave ({self.frequency} Hz)"
        self.waveform_canvas.plot_waveform(edited_waveform, sr, total_duration, title)

    # ===== PLAY SOUND FUNCTIONS =====
    def play_square_sound(self):
        self.current_waveform_type = "square"
        self.update_waveform()
        
        waveform, sr, total_duration = generate_square_wave(
            self.frequency, 
            self.amplitude
        )
        
        edited_waveform = waveform.copy()
    
        # Apply low-pass filter
        if self.lowpass_checkbox.isChecked():
            edited_waveform = apply_lowpass_filter(edited_waveform, cutoff=self.cutoff, sample_rate=sr)
            
        # Apply distortion filter if checked
        if self.distortion_checkbox.isChecked():
            edited_waveform = apply_distortion(edited_waveform)
            
        # Apply noise filter if checked
        if self.noise_checkbox.isChecked():
            edited_waveform = generate_noise(amplitude=self.amplitude)
            
        # Apply vibrato filter if checked
        if self.vibrato_checkbox.isChecked():
            edited_waveform, sr, total_duration = generate_vibrato(frequency=self.frequency, amplitude=self.amplitude)
        
        # If the checkbox is checked, convert the waveform to 8-bit
        if self.bit_depth_checkbox.isChecked():
            edited_waveform = convert_to_bit_depth(edited_waveform, bit_depth=8)
            # Properly convert from uint8 (0 to 255) to int16 (-32768 to 32767)
            edited_waveform = (edited_waveform.astype(np.int16) - 128) * 256

        # Play the generated sound
        sd.play(edited_waveform, sr)

        # Save the sound file with correct duration
        sf.write("generated_audio.wav", edited_waveform, sr)
        print("Waveform Stats:")
        print("Min:", np.min(waveform), "Max:", np.max(waveform), "Mean:", np.mean(waveform))
        print("Sample Rate:", sr, "Duration:", total_duration)
        
    def play_sawtooth_sound(self):
        self.current_waveform_type = "sawtooth"
        self.update_waveform()
        
        waveform, sr, total_duration = generate_sawtooth_wave(
            self.frequency, 
            self.amplitude
        )
        
        edited_waveform = waveform.copy()
    
        # Apply low-pass filter
        if self.lowpass_checkbox.isChecked():
            edited_waveform = apply_lowpass_filter(edited_waveform, cutoff=self.cutoff, sample_rate=sr)
            
        # Apply distortion filter if checked
        if self.distortion_checkbox.isChecked():
            edited_waveform = apply_distortion(edited_waveform)
            
        # Apply noise filter if checked
        if self.noise_checkbox.isChecked():
            edited_waveform = generate_noise(amplitude=self.amplitude)
            
        # Apply vibrato filter if checked
        if self.vibrato_checkbox.isChecked():
            edited_waveform, sr, total_duration = generate_vibrato(frequency=self.frequency, amplitude=self.amplitude)
        
        # If the checkbox is checked, convert the waveform to 8-bit
        if self.bit_depth_checkbox.isChecked():
            edited_waveform = convert_to_bit_depth(edited_waveform, bit_depth=8)
            # Properly convert from uint8 (0 to 255) to int16 (-32768 to 32767)
            edited_waveform = (edited_waveform.astype(np.int16) - 128) * 256

        # Play the generated sound
        sd.play(edited_waveform, sr)

        # Save the sound file with correct duration
        sf.write("generated_audio.wav", edited_waveform, sr)
        print("Waveform Stats:")
        print("Min:", np.min(waveform), "Max:", np.max(waveform), "Mean:", np.mean(waveform))
        print("Sample Rate:", sr, "Duration:", total_duration)
        
    def play_sine_sound(self):
        self.current_waveform_type = "sine"
        self.update_waveform()
        
        waveform, sr, total_duration = generate_sine_wave(
            self.frequency, 
            self.amplitude, 
            attack=self.attack, 
            decay=self.decay, 
            sustain=self.sustain, 
            release=self.release
        )
        
        edited_waveform = waveform.copy()
    
        # Apply low-pass filter
        if self.lowpass_checkbox.isChecked():
            edited_waveform = apply_lowpass_filter(edited_waveform, cutoff=self.cutoff, sample_rate=sr)
            
        # Apply distortion filter if checked
        if self.distortion_checkbox.isChecked():
            edited_waveform = apply_distortion(edited_waveform)
            
        # Apply noise filter if checked
        if self.noise_checkbox.isChecked():
            edited_waveform = generate_noise(amplitude=self.amplitude)
            
        # Apply vibrato filter if checked
        if self.vibrato_checkbox.isChecked():
            edited_waveform, sr, total_duration = generate_vibrato(frequency=self.frequency, amplitude=self.amplitude)
        
        # If the checkbox is checked, convert the waveform to 8-bit
        if self.bit_depth_checkbox.isChecked():
            edited_waveform = convert_to_bit_depth(edited_waveform, bit_depth=8)
            # Properly convert from uint8 (0 to 255) to int16 (-32768 to 32767)
            edited_waveform = (edited_waveform.astype(np.int16) - 128) * 256

        # Play the generated sound
        sd.play(edited_waveform, sr)

        # Save the sound file with correct duration
        sf.write("generated_audio.wav", edited_waveform, sr)
        # print("Waveform Stats:")
        # print("Min:", np.min(waveform), "Max:", np.max(waveform), "Mean:", np.mean(waveform))
        # print("Sample Rate:", sr, "Duration:", total_duration)


# Run Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioApp()
    window.show()
    sys.exit(app.exec())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.fft import fft, fftfreq, fftshift
from scipy.special import erf

# Manual sinusoidal decomposition
def sinusoidal_decomposition(signal, t, freqs):
    T = t[-1] - t[0]
    dt = t[1] - t[0]
    cosine_weights = []
    sine_weights = []
    for f in freqs:
        cos_comp = np.trapezoid(signal * np.cos(2 * np.pi * f * t), dx=dt)
        sin_comp = np.trapezoid(signal * np.sin(2 * np.pi * f * t), dx=dt)
        cosine_weights.append(cos_comp)
        sine_weights.append(sin_comp)
    cosine_weights = np.array(cosine_weights)
    sine_weights = np.array(sine_weights)
    magnitude = np.sqrt(cosine_weights**2 + sine_weights**2)
    return magnitude, cosine_weights, sine_weights

# Initial parameters
init_freq = 1.0 # GHz
init_risetime = 50     # ps
init_samplerate = 10 # GHz
init_timeshift = 0     # seconds
num_cycles = 20

# Figure and axes
fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=False)
fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.5)
(ax1, ax2, ax3, ax4, ax5) = axes

# Slider figure
fig_sliders, ax_sliders = plt.subplots(figsize=(6, 3))
fig_sliders.canvas.manager.set_window_title("Control Panel")
plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.25)
ax_sliders.axis('off')  # No visible axes

# Create individual slider axes
slider_axes = {
    'freq':       fig_sliders.add_axes([0.2, 0.8, 0.7, 0.05]),
    'risetime':   fig_sliders.add_axes([0.2, 0.65, 0.7, 0.05]),
    'samplerate': fig_sliders.add_axes([0.2, 0.5, 0.7, 0.05]),
    'timeshift':  fig_sliders.add_axes([0.2, 0.35, 0.7, 0.05])
}

# Create sliders
s_frequency   = Slider(slider_axes['freq'], 'Signal Freq (GHz)',     1e-3, 10, valinit=init_freq, valstep=1e-3)
s_risetime    = Slider(slider_axes['risetime'], 'Rise Time (ps)',   5, 200, valinit=init_risetime)
s_samplerate  = Slider(slider_axes['samplerate'], 'Sample Rate (GHz)', 1e-3, 40, valinit=init_samplerate, valstep=1.0)
s_timeshift   = Slider(slider_axes['timeshift'], 'Time Shift (ns)', -2, 2, valinit=0, valstep=0.01)

def update(val):
    signal_freq = s_frequency.val*1e9  # Convert GHz to Hz
    T = 1 / signal_freq
    rise_time = s_risetime.val * 1e-12
    sampling_rate = s_samplerate.val*1e9  # Convert GHz to Hz
    dt = 1 / sampling_rate
    time_shift = s_timeshift.val * 1e-9
    t = np.arange(0, num_cycles * T, dt) + time_shift

    signal = np.zeros_like(t)
    for n in range(2 * num_cycles):
        edge_time = n * (T / 2)
        sign = 1 if n % 2 == 0 else -1
        signal += 0.5 * (1 + sign * erf((t - edge_time) / (rise_time / 2.0)))

    ax1.cla()
    ax1.plot(t * 1e9, signal, 'o-', markersize=2)
    ax1.set_title("Time-Domain Signal")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Voltage")
    ax1.grid(True)

    N = len(t)
    freqs_fft = fftshift(fftfreq(N, dt))
    spectrum_fft = fftshift(np.abs(fft(signal)))
    ax2.cla()
    ax2.plot(freqs_fft / 1e9, 20 * np.log10(spectrum_fft / np.max(spectrum_fft)))
    ax2.set_title("FFT Spectrum")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_xlim(0, sampling_rate / 2 / 1e9)
    ax2.grid(True)

    freqs_manual = np.linspace(0, 5e9, 400)
    mag_manual, cos_weights, sin_weights = sinusoidal_decomposition(signal, t, freqs_manual)
    ax3.cla()
    ax3.plot(freqs_manual / 1e9, 20 * np.log10(mag_manual / np.max(mag_manual)))
    ax3.set_title("Manual Spectrum (Magnitude)")
    ax3.set_xlabel("Frequency (GHz)")
    ax3.set_ylabel("Magnitude (dB)")
    ax3.grid(True)

    ax4.cla()
    ax4.plot(freqs_manual / 1e9, cos_weights)
    ax4.set_title("Cosine Weights")
    ax4.set_ylabel("Amplitude")
    ax4.grid(True)

    ax5.cla()
    ax5.plot(freqs_manual / 1e9, sin_weights)
    ax5.set_title("Sine Weights")
    ax5.set_xlabel("Frequency (GHz)")
    ax5.set_ylabel("Amplitude")
    ax5.grid(True)

    fig.canvas.draw_idle()

# Connect sliders
for s in [s_frequency, s_risetime, s_samplerate, s_timeshift]:
    s.on_changed(update)

update(None)
plt.show()

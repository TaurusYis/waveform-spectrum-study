import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import erf
from scipy.fft import fft, fftfreq, fftshift

# ----- Sinusoidal Decomposition -----
def sinusoidal_decomposition(signal, t, freq_list):
    magnitudes = []
    T = t[-1] - t[0]
    for f in freq_list:
        cos_component = np.cos(2 * np.pi * f * t)
        sin_component = np.sin(2 * np.pi * f * t)
        C = (2 / T) * np.trapezoid(signal * cos_component, t)
        S = (2 / T) * np.trapezoid(signal * sin_component, t)
        A = np.sqrt(C**2 + S**2)
        magnitudes.append(A)
    return np.array(magnitudes)

# ----- Plotting Function -----
def update(val):
    bit_rate = s_bitrate.val * 1e9
    rise_time = s_risetime.val * 1e-12
    sampling_rate = s_samplerate.val * 1e9
    bit_period = 1 / bit_rate
    dt = 1 / sampling_rate
    t = np.arange(0, 10 * bit_period, dt)

    signal = np.zeros_like(t)
    for n in range(0, int(len(t) * dt / (bit_period / 2))):
        edge_time = n * (bit_period / 2)
        sign = 1 if n % 2 == 0 else -1
        signal += 0.5 * (1 + sign * erf((t - edge_time) / (rise_time / 2.0)))

    # Time domain
    ax1.cla()
    ax1.plot(t * 1e9, signal)
    ax1.set_title("DDR Signal (Time Domain)")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Voltage")
    ax1.grid(True)

    # FFT
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

    # Sinusoidal Decomposition
    freqs_manual = np.linspace(0, 5e9, 300)
    spectrum_manual = sinusoidal_decomposition(signal, t, freqs_manual)
    ax3.cla()
    ax3.plot(freqs_manual / 1e9, 20 * np.log10(spectrum_manual / np.max(spectrum_manual)))
    ax3.set_title("Sinusoidal Decomposition Spectrum")
    ax3.set_xlabel("Frequency (GHz)")
    ax3.set_ylabel("Magnitude (dB)")
    ax3.grid(True)

    fig.canvas.draw_idle()

# ----- Figure and Layout -----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
plt.subplots_adjust(left=0.1, bottom=0.3)

# ----- Sliders -----
axcolor = 'lightgoldenrodyellow'
ax_bitrate = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor=axcolor)
ax_risetime = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
ax_samplerate = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)

s_bitrate = Slider(ax_bitrate, 'Bit Rate (Gbps)', 0.5, 10.0, valinit=1.0, valstep=0.1)
s_risetime = Slider(ax_risetime, 'Rise Time (ps)', 10, 200, valinit=50, valstep=1)
s_samplerate = Slider(ax_samplerate, 'Sample Rate (Gsps)', 2.0, 100.0, valinit=10.0, valstep=0.5)

# ----- Link Sliders to Update -----
s_bitrate.on_changed(update)
s_risetime.on_changed(update)
s_samplerate.on_changed(update)

# ----- Initial Plot -----
update(None)
plt.show()

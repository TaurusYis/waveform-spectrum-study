import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import erf
from scipy.fft import fft, fftfreq, fftshift

# ----- Sinusoidal Decomposition -----
def sinusoidal_decomposition(signal, t, freq_list):
    magnitudes = []
    T = t[-1] - t[0]
    dt = t[1] - t[0]
    for f in freq_list:
        cos_component = np.cos(2 * np.pi * f * t)
        sin_component = np.sin(2 * np.pi * f * t)
        C = (2 / T) * np.sum(signal * cos_component) * dt
        S = (2 / T) * np.sum(signal * sin_component) * dt
        A = np.sqrt(C**2 + S**2)
        magnitudes.append(A)
    return np.array(magnitudes)

# ----- Plotting Function -----
def update(val):
    signal_freq = s_frequency.val  # in Hz
    rise_time = s_risetime.val * 1e-12  # ps to s
    sampling_rate = s_samplerate.val  # already in Hz

    T = 1 / signal_freq
    dt = 1 / sampling_rate
    t = np.arange(0, 10 * T, dt)

    # Generate square wave with finite rise/fall time using erf
    signal = np.zeros_like(t)
    for n in range(20):  # 10 cycles, 2 transitions per cycle
        edge_time = n * (T / 2)
        sign = 1 if n % 2 == 0 else -1
        signal += 0.5 * (1 + sign * erf((t - edge_time) / (rise_time / 2.0)))

    # Plot time domain
    ax1.cla()
    ax1.plot(t * 1e9, signal)
    ax1.set_title("Signal (Time Domain)")
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

    # Manual decomposition
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
ax_freq = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor=axcolor)
ax_risetime = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor=axcolor)
ax_samplerate = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)

s_frequency = Slider(ax_freq, 'Signal Freq (Hz)', 1e6, 10e9, valinit=1e9, valstep=1e6)
s_risetime = Slider(ax_risetime, 'Rise Time (ps)', 10, 200, valinit=50, valstep=1)
s_samplerate = Slider(ax_samplerate, 'Sample Rate (Hz)', 2e9, 40e9, valinit=10e9, valstep=1e9)

# ----- Link Sliders to Update -----
s_frequency.on_changed(update)
s_risetime.on_changed(update)
s_samplerate.on_changed(update)

# ----- Initial Plot -----
update(None)
plt.show()

"""
Waveform shape & Instantaneous Frequency
========================================
Here we explore how the instantaneous frequency of a signal is related to its
waveform shape and how we can directly compare waveform shapes using phase
alignment

"""

#%%
# We will start with some imports emd and by simulating a very simple
# stationary sine wave signal.

import emd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#%%
# Linear & Non-linear Systems
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%
# In this tutorial, we're going to explore how the instaneous frequency of an
# oscillatory signal can represent its waveform shape. To do this, we're going
# create a sine-wave simulation and modulate by a linear and a non-linear
# equation. The linear equation simply scales the signal by a defined factor.
# The non-linear equation also scales the signal but, crucially, has an extra
# term which distorts the waveform of the oscillation such that it becomes
# non-sinusoidal.
#
# These equations are implemented as functions below. The equations themselves
# were defined in equations 50.24 and 50.25 in section 50-6 of Feynman's
# Lectures of Physics.


def linear_system(x, K):
    """ A linear system which scales a signal by a factor"""
    return K * x


def nonlinear_system(x, K, eta=.43, power=2):
    """ A non-linear system which scales a signal by a factor introduces a
    waveform distortion"""
    return K * (x + eta * (x ** power))


#%%
# A simple sine-wave
#^^^^^^^^^^^^^^^^^^^

#%%
# We will first apply our linear and non-linear equations to a very simple
# pure-tone oscillation. We define some values below and create a 10 second
# signal which oscillates at 2Hz.

seconds = 10
f = 2
sample_rate = 512
emd.spectra.frequency_stats
t = np.linspace(0, seconds, seconds*sample_rate)
x = np.cos(2*np.pi*f*t)


#%%
# We then modulate our signal ``x`` by the linear and nonlinear systems.

K = .5
power = 2

x_linear = linear_system(x, K)
x_nonlinear = nonlinear_system(x, K)

# Create a summary plot
plt.figure(figsize=(10, 4))
plt.plot(t, x, 'k:')
plt.plot(t, x_linear)
plt.plot(t, x_nonlinear)
plt.xlim(0, 3)
plt.xlabel('Time (seconds)')
plt.legend(['Original', 'Linear', 'Non-linear'])

#%%
# We can see that the output of the linear system returns a scaled sinusoid
# whilst the nonlinear system outputs a distorted wave. By eye, we can see that
# the non-linear signal has a sharper peak and a wider trough than the linear
# system. The next section is going to quantify this distortion using
# instantanous frequency.
#
# Firstly, we compute the EMD of the linear system using the ``emd.sift.sift`` with
# default argumnts.

# Compute EMD
imf_linear = emd.sift.sift(x_linear)

# Visualise the IMFs
emd.plotting.plot_imfs(imf_linear[:sample_rate*4, :], cmap=True, scale_y=True)

#%%
# This is an easy decomposition as we haven't added any noise to the signal.
# The oscillation is captured completed by the first component whilst the
# second component contains a very small residual.
#
# Next we compute the EMD for the non-linear system

# Compute EMD
imf_nonlinear = emd.sift.sift(x_nonlinear)

# Visualise the IMFs
emd.plotting.plot_imfs(imf_nonlinear[:sample_rate*4, :], cmap=True, scale_y=True)

#%%
# As with the linear system, this is an easy decomposition without any noise.
# The oscillatory signal is captured within the first component without further
# distorting the waveform shape. The residual contains a near-constant mean
# term. This is as the non-linear system makes the peaks larger and the troughs
# smaller which shifts the mean of the signal away from zero. This effect is
# often called rectification.
#
# Next, we compute the instantanous frequency metrics from our linear and
# non-linear IMFs.

IP_linear, IF_linear, IA_linear = emd.spectra.frequency_stats(imf_linear, sample_rate, 'nht')
IP_nonlinear, IF_nonlinear, IA_nonlinear = emd.spectra.frequency_stats(imf_nonlinear, sample_rate, 'nht')

#%%
# We can now start to look at how a non-sinusoidal waveform is represented in
# frequency. We will compare the EMD instantnaous frequency perspective with a
# standard frequency andlysis based on the Fourier transform.
#
# We compute the Hilbert-Huang transform from the IMF frequency metrics and
# Welch's Periodogram from the raw data before creating a summary plot.

# Welch's Periodogram
f, pxx_linear = signal.welch(x_linear, fs=sample_rate, nperseg=2048)
f, pxx_nonlinear = signal.welch(x_nonlinear, fs=sample_rate, nperseg=2048)

# Hilbert-Huang transform
edges, centres = emd.spectra.define_hist_bins(0, 20, 64)
spec_linear = emd.spectra.hilberthuang_1d(IF_linear, IA_linear, edges) / len(x)
spec_nonlinear = emd.spectra.hilberthuang_1d(IF_nonlinear, IA_nonlinear, edges) / len(x)

# Summary figure
plt.figure()
plt.subplot(121)
plt.plot(f, pxx_linear)
plt.plot(f, pxx_nonlinear)
plt.title("Welch's Periodogram")
plt.xlim(0, 20)
plt.xticks(np.arange(10)*2)
plt.grid(True)
plt.xlabel('Frequency (Hz)')

plt.subplot(122)
plt.plot(centres, spec_linear[:, 0])
plt.plot(centres, spec_nonlinear[:, 0])
plt.xticks(np.arange(10)*2)
plt.grid(True)
plt.title("Hilbert-Huang Transform")
plt.legend(['Linear System', 'Nonlinear System'])
plt.xlabel('Frequency (Hz)')

#%%
# Both the Welch and Hilbert-Huang transform show a clear 2Hz peak for the
# linear system but differ in how the represent the non-linear system. Welch's
# Periodogram introduces a harmonic component at 4Hz whereas the Hilbert-Huang
# transform simply widens the existing 2Hz peak.
#
# Why would a non-sinsuoidal signal lead to a wider spectral peak in the
# Hilbert-Huang transform? To get some intuition about this, we will plot the
# Hilbert-Huang spectra alongside the instantaneous frequency traces for the
# linear and non-linear systems.

plt.figure(figsize=(12, 8))
plt.axes([.1, .1, .2, .8])
plt.plot(spec_linear[:, 0], centres)
plt.plot(spec_nonlinear[:, 0], centres)
plt.ylim(0, 10)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Power')
plt.title('HHT')

plt.axes([.32, .1, .65, .8])
plt.plot(t, IF_linear[:, 0])
plt.plot(t, IF_nonlinear[:, 0])
plt.ylim(0, 10)
plt.xlim(0, 5)
plt.legend(['Linear system', 'Nonlinear-system'])
plt.title('Instantaneous Frequency')
plt.xlabel('Time (seconds)')

#%%
# We see that the linear system has a constant instantaneous frequency which
# doesn't vary over time. When this constant instantanous frequency is fed into
# the Hilbert-Huang transform it concentrates all the power into a sharp peak
# which looks similar to Welch's periodogram.
#
# In contrast, the instantanous frequency of the non-linear system does change
# over time. In fact, it seems to be oscillating between values aorund 2Hz (The
# IF variability is actually 2+/- the value for eta defined in the function
# above). When this variable instantaneous frequency is fed into the
# Hilbert-Huang transform, it spreads the power out within this same range.
#
# If you re-run this analysis with a small value of eta in the nonlinear_system
# function you will see that the instantaneous frequency here varies within a
# smaller range and the peak in the Hilbert-Huang transform gets sharper again.
#
# The variability in instantaneous frequency reflects the waveform shape
# distortions introduced by the non-linear system. We can see this by taking a
# look at the original waveform and the instantnaous freuqencies alongside each
# other.

plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(t, x_linear)
plt.plot(t, x_nonlinear)
plt.xlim(0, 3)
plt.subplot(212)
plt.plot(t, IF_linear[:, 0])
plt.plot(t, IF_nonlinear[:, 0])
plt.xlim(0, 3)
plt.ylim(0, 4)

#%%
# The peaks in instantaneous frequency co-incide with the peaks of the raw
# signal, whilst the lowest instantaneous frequency values occur around the
# trough. This reflects how quickly the oscillation is progressing at each
# point in the cycle. The linear system progresses at a uniform rate throughout
# each cycle and therefore has a constant instantaneous frequency. In
# contrast, the sharp peaks and wide troughs of the non-linear signal can be
# interpreted as the cycle processing more quickly and slowly at the peak and
# trough respectively. The instantnaous frequency tracks this at the full
# sample rate of the data showing high frequnecies around the sharp peaks and
# low frequencies around the slow troughs.

#%%
# A dynamic oscillation with noise
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%
# Unfortunately most signals are more complex than our sine-wave above! Here we
# apply the same analysis as above to a dynamic & noisy signal. The signal
# dynamics make the signal more interesting but also introduce some challenges
# for waveform shape analyses, we will explore what these are and how
# phase-alignment can be useful to overcome them.
#
# First we generate a dynamic oscillation using direct pole-placement to create
# an autoregressive model with a peak around 12Hz. We then pass the dynamic
# oscillation through our linear and non-linear systems as above. Finally we
# add some white noise.

peak_freq = 12
sample_rate = 512
seconds = 60
noise_std = None
x = emd.utils.ar_simulate(peak_freq, sample_rate, seconds, noise_std=noise_std, random_seed=42, r=.99)
x = x * 1e-5
t = np.linspace(0, seconds, seconds*sample_rate)

x_linear = linear_system(x, K=1) + np.random.randn(len(t), 1)*5e-2
x_nonlinear = nonlinear_system(x, K=1, eta=2) + np.random.randn(len(t), 1)*5e-2

#%%
# We compute our IMFs using the mask_sift with default parameters. First on the linear system.

# Compute IMFs
imf_linear = emd.sift.mask_sift(x_linear)

# Visualise IMFs
emd.plotting.plot_imfs(imf_linear[:sample_rate*4, :], cmap=True, scale_y=True)

#%%
# The oscillation is isolated into IMF-3. The remaining IMFs comtain low
# magnitude noise. Next we run the same on the non-linear system.

# Compute IMFs
imf_nonlinear = emd.sift.mask_sift(x_nonlinear)

# Visualise IMFs
emd.plotting.plot_imfs(imf_nonlinear[:sample_rate*4, :], cmap=True, scale_y=True)

#%%
# Again the oscillatory component is isolated into IMF-3. Next we compute the
# instantanous frequency metrics for the linear and nonlinear IMFs using the
# Normalise Hilbert Transform.

IP_linear, IF_linear, IA_linear = emd.spectra.frequency_stats(imf_linear, sample_rate, 'nht')
IP_nonlinear, IF_nonlinear, IA_nonlinear = emd.spectra.frequency_stats(imf_nonlinear, sample_rate, 'nht')

#%%
# We next compare the spectral content of the signal using the EMD based
# Hilbert-Huang transform and the Fourier based Welch's Periodogram.

# Welch's Periodogram
f, pxx_linear = signal.welch(x_linear[:, 0], fs=sample_rate, nperseg=2048)
f, pxx_nonlinear = signal.welch(x_nonlinear[:, 0], fs=sample_rate, nperseg=2048)

# Hilbert-Huang Transform
edges, centres = emd.spectra.define_hist_bins(0, 40, 64)
spec_linear = emd.spectra.hilberthuang_1d(IF_linear, IA_linear, edges, mode='amplitude')
spec_nonlinear = emd.spectra.hilberthuang_1d(IF_nonlinear, IA_nonlinear, edges, mode='amplitude')

# Summary figure
plt.figure()
plt.subplot(121)
plt.plot(f, pxx_linear)
plt.plot(f, pxx_nonlinear)
plt.title("Welch's Periodogram")
plt.xlim(0, 40)
plt.subplot(122)
plt.plot(centres, spec_linear[:, 2])
plt.plot(centres, spec_nonlinear[:, 2])
plt.title("Hilbert-Huang Transform")
plt.legend(['Linear System', 'Nonlinear System'])

#%%
# As with the simple sinusoidal signal in the first section. We see that the
# non-sinsusoidal waveform introduced by the nonlinear sytemintroduces a
# harmonic into Welch's Periodogram and widens the 12Hz peak of the
# Hilbert-Huang transform.
#
# We can plot the waveform and instantanous frequency alongside each other to
# try and see how the shape might be affecting instantanous frequency.

plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(t, imf_linear[:, 2])
plt.plot(t, imf_nonlinear[:, 2])
plt.xlim(0, 3)
plt.subplot(212)
plt.plot(t, IF_linear[:, 2])
plt.plot(t, IF_nonlinear[:, 2])
plt.xlim(0, 3)
plt.ylim(0, 25)
plt.ylabel('Instantaneous\nFrequency (Hz)')
plt.xlabel('Time (seconds)')
plt.legend(['Linear System', 'Nonlinear System'])

#%%
# In contrast to the simple sinusoidal case, this plot looks very noisy. The
# instantaneous frequency estimates are very volitile in parts of the signal
# with low amplitude (such as 1-1.75 seconds). If we concentrate on clean parts
# of the signal (say 0-0.5 seconds) we can perhaps see a suggestion that the
# non-linear instantnaous frequency is changing more than the linear one but it
# is perhaps hard to tell from this alone.
#
# We can try to clean up the analysis by contentrating on oscillatory cycles
# which have a well formed phase and an amplitude above a specified threshold.
# We extract these cycles using the ``emd.cycles.get_cycle_inds`` function with
# a defined mask based on instantanous amplitude.

cycles_linear = emd.cycles.get_cycle_inds(IP_linear, return_good=True, mask=IA_linear[:, 2] > .05)
cycles_nonlinear = emd.cycles.get_cycle_inds(IP_nonlinear, return_good=True, mask=IA_nonlinear[:, 2] > .05)

#%%
# We can now plot just the 'good' cycles in the analysis locked to the ascending zero-crossing.

plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for ii in range(1, cycles_linear.max()+1):
    inds = cycles_linear[:, 2] == ii
    ax1.plot(imf_linear[inds, 2])
    ax2.plot(IF_linear[inds, 2])

plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for ii in range(1, cycles_nonlinear.max()+1):
    inds = cycles_nonlinear[:, 2] == ii
    ax1.plot(imf_nonlinear[inds, 2])
    ax2.plot(IF_nonlinear[inds, 2])


#%%

pa_linear = emd.cycles.phase_align(IP_linear[:, 2, None], IF_linear[:, 2], cycles=cycles_linear[:, 2, None])
pa_nonlinear = emd.cycles.phase_align(IP_nonlinear[:, 2, None], IF_nonlinear[:, 2], cycles=cycles_nonlinear[:, 2, None])

plt.figure()
plt.plot(pa_linear.mean(axis=1))
plt.plot(pa_nonlinear.mean(axis=1))
plt.ylim(10, 16)
plt.legend(['Linear System', 'Nonlinear system'])

"""
Describing waveform shape with Instantaneous Frequency and phase-alignment
==========================================================================
Here we explore how the instantaneous frequency of a signal is related to its
waveform shape and how we can directly compare waveform shapes using phase
alignment

"""

#%%
# Simulating a noisy signal
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Firstly we will import emd and simulate a signal

import emd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Feynman 50-6 eqn 50.25
# http://www.feynmanlectures.caltech.edu/I_50.html


seconds = 10
f = 2
sample_rate = 512
emd.spectra.frequency_stats
t = np.linspace(0, seconds, seconds*sample_rate)
x = np.cos(2*np.pi*f*t)


def linear_system(x, K):
    return K * x


def nonlinear_system(x, K, eta=.43, power=2):
    return K * (x + eta * (x ** power))


K = .5
power = 2

x_linear = linear_system(x, K)
x_nonlinear = nonlinear_system(x, K)

plt.figure(figsize=(10, 4))
plt.plot(t, x, 'k:')
plt.plot(t, x_linear)
plt.plot(t, x_nonlinear)
plt.xlim(0, 3)
plt.legend(['Original', 'Linear', 'Non-linear'])

#%%

imf_linear = emd.sift.sift(x_linear)

# Visualise the IMFs
emd.plotting.plot_imfs(imf_linear[:sample_rate*4, :], cmap=True, scale_y=True)

#%%

imf_nonlinear = emd.sift.sift(x_nonlinear)

# Visualise the IMFs
emd.plotting.plot_imfs(imf_nonlinear[:sample_rate*4, :], cmap=True, scale_y=True)


#%%

IP_linear, IF_linear, IA_linear = emd.spectra.frequency_stats(imf_linear, sample_rate, 'nht')
IP_nonlinear, IF_nonlinear, IA_nonlinear = emd.spectra.frequency_stats(imf_nonlinear, sample_rate, 'nht')

#%%

edges, centres = emd.spectra.define_hist_bins(0, 20, 64)
spec_linear = emd.spectra.hilberthuang_1d(IF_linear, IA_linear, edges) / len(x)
spec_nonlinear = emd.spectra.hilberthuang_1d(IF_nonlinear, IA_nonlinear, edges) / len(x)

f, pxx_linear = signal.welch(x_linear, fs=sample_rate, nperseg=2048)
f, pxx_nonlinear = signal.welch(x_nonlinear, fs=sample_rate, nperseg=2048)

plt.figure()
plt.subplot(121)
plt.plot(f, pxx_linear)
plt.plot(f, pxx_nonlinear)
plt.title("Welch's Periodogram")
plt.xlim(0, 20)
plt.subplot(122)
plt.plot(centres, spec_linear[:, 0])
plt.plot(centres, spec_nonlinear[:, 0])
plt.title("Hilbert-Huang Transform")
plt.legend(['Linear System', 'Nonlinear System'])


#%%

plt.figure(figsize=(12, 8))
plt.axes([.1, .1, .2, .8])
plt.plot(spec_linear[:, 0], centres)
plt.plot(spec_nonlinear[:, 0], centres)
plt.ylim(0, 10)

plt.axes([.32, .1, .65, .8])
plt.plot(t, IF_linear[:, 0])
plt.plot(t, IF_nonlinear[:, 0])
plt.ylim(0, 10)
plt.xlim(0, 5)


#%%

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

peak_freq = 12
sample_rate = 512
seconds = 60
noise_std = None
x = emd.utils.ar_simulate(peak_freq, sample_rate, seconds, noise_std=noise_std, random_seed=42, r=.99)
x = x * 1e-5
t = np.linspace(0, seconds, seconds*sample_rate)

x_linear = linear_system(x, K=1) + np.random.randn(len(t), 1)*5e-2
x_nonlinear = nonlinear_system(x, K=1, eta=2) + np.random.randn(len(t), 1)*5e-2


imf_linear = emd.sift.mask_sift(x_linear)
emd.plotting.plot_imfs(imf_linear[:sample_rate*4, :], cmap=True, scale_y=True)

imf_nonlinear = emd.sift.mask_sift(x_nonlinear)
emd.plotting.plot_imfs(imf_nonlinear[:sample_rate*4, :], cmap=True, scale_y=True)


IP_linear, IF_linear, IA_linear = emd.spectra.frequency_stats(imf_linear, sample_rate, 'nht')
IP_nonlinear, IF_nonlinear, IA_nonlinear = emd.spectra.frequency_stats(imf_nonlinear, sample_rate, 'nht')

#%%

edges, centres = emd.spectra.define_hist_bins(0, 40, 64)
spec_linear = emd.spectra.hilberthuang_1d(IF_linear, IA_linear, edges) / len(x)
spec_nonlinear = emd.spectra.hilberthuang_1d(IF_nonlinear, IA_nonlinear, edges) / len(x)

print(spec_linear[:, 2].sum())
print(spec_nonlinear[:, 2].sum())

f, pxx_linear = signal.welch(x_linear[:, 0], fs=sample_rate, nperseg=2048)
f, pxx_nonlinear = signal.welch(x_nonlinear[:, 0], fs=sample_rate, nperseg=2048)

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


cycles_linear = emd.cycles.get_cycle_inds(IP_linear, return_good=True, mask=IA_linear[:, 2] > .05)
cycles_nonlinear = emd.cycles.get_cycle_inds(IP_nonlinear, return_good=True, mask=IA_nonlinear[:, 2] > .05)


#%%


plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(t, imf_nonlinear[:, 2])
plt.xlim(0, 3)
plt.subplot(212)
plt.plot(t, IF_nonlinear[:, 2])
plt.xlim(0, 3)


#%%

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

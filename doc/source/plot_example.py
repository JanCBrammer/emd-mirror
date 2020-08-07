import emd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Define and simulate a simple signal
peak_freq = 12
sample_rate = 512
seconds = 20
noise_std = .5
x = emd.utils.ar_simulate(peak_freq, sample_rate, seconds, noise_std=noise_std, random_seed=42, r=.99) * 1e-4
t = np.linspace(0, seconds, seconds*sample_rate)

x = x + (np.cos(2*np.pi*.5*t)[:,None]+1)*np.sin(2*np.pi*2.5*t)[:,None] / 2

y = 2*np.sin(2*np.pi*85*t)[:,None]
y[x<1] = 0

x = x + y

imf = emd.sift.mask_sift(x,max_imfs=7, mask_freqs=.25,mask_amp_mode='ratio_sig')
IP, IF, IA = emd.spectra.frequency_stats(imf, sample_rate, 'nht')

def mask_sift_second_layer(IA, masks, config={}):
    imf2 = np.zeros((IA.shape[0], IA.shape[1], config['max_imfs']))
    for ii in range(IA.shape[1]):
        config['mask_freqs'] = masks[ii:]
        tmp = emd.sift.mask_sift(IA[:, ii], **config)
        imf2[:, ii, :tmp.shape[1]] = tmp
    return imf2


# Define sift parameters for the second level
masks = np.array([25/2**ii for ii in range(12)])/sample_rate
config = emd.sift.get_config('mask_sift')
config['mask_amp_mode'] = 'ratio_sig'
config['mask_amp'] = 2
config['max_imfs'] = 5
config['imf_opts/sd_thresh'] = 0.05
config['envelope_opts/interp_method'] = 'mono_pchip'

# Sift the first 5 first level IMFs
imf2 = mask_sift_second_layer(IA, masks, config=config)
IP2, IF2, IA2 = emd.spectra.frequency_stats(imf2, sample_rate, 'nht')


# Carrier frequency histogram definition
edges, bins = emd.spectra.define_hist_bins(1, 200, 128, 'log')
# AM frequency histogram definition
edges2, bins2 = emd.spectra.define_hist_bins(1e-2, 32, 64, 'log')

# Compute the 1d Hilbert-Huang transform (power over carrier frequency)
spec = emd.spectra.hilberthuang_1d(IF, IA, edges)

# Compute the 2d Hilbert-Huang transform (power over time x carrier frequency)
hht = emd.spectra.hilberthuang(IF, IA, edges)
shht = ndimage.gaussian_filter(hht, 2)

# Compute the 3d Holospectrum transform (power over time x carrier frequency x AM frequency)
# Here we return the time averaged Holospectrum (power over carrier frequency x AM frequency)
holo = emd.spectra.holospectrum(IF[:, :], IF2[:, :, :], IA2[:, :, :], edges, edges2)

emd.plotting.plot_imfs(imf[256:sample_rate*3+256,:],cmap=True,scale_y=True)

plt.figure()
plt.pcolormesh(t,edges,shht,cmap='hot_r')
plt.gca().set_yscale('log')

plt.figure()
plt.pcolormesh(edges2,edges,holo.T,cmap='hot_r')
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')

plt.show()




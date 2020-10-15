---
title: "EMD: Empirical Mode Decomposition, Hilbert-Huang Transform and Holospectrum analyses in Python"
tags:
  - Python
  - Time-series
  - Non-linear
  - Dynamics
authors:
  - name: A. N. Author (to be completed)
    affiliation: 1
affiliations:
  - name: OHBA
    index: 1
date: 12 October 2020
bibliography: paper.bib
---

# Summary

The Empirical Mode Decomposition ([EMD](https://emd.readthedocs.io/en/latest/))
package contains Python (>3.5) functions for analysis of non-linear and
non-stationary oscillatory time series. `EMD` implements a family of sifting
agorithms, instantaneous frequency transformations, power spectrum construction
and single-cycle feature extraction. Many natural signals contain non-linear or
non-sinusoidal features that change dynamically over time. These complex and
dynamic features are often of analytic interest but can confound standard
analyses such as the Fourier trasnform that assume linear and stationary
signals. The Empirical Mode Decomposition is defined by the sift-algorithm; a
data-adaptive decomposition technique that separates a signal into a set of
physically intpretable Intrinsic Mode Functions (IMFs) that permit well behaved
Hilbert transforms [@Huang1998]. Crucially, this decomposition work on the
local features of the dataset and therefore the IMFs can retain the non-linear
and non-stationary characteristics of the signal.

# Package Features

The sift algorithm is implemented in the `emd.sift` module, including the
classic sift (`emd.sift.sift`), the Ensemble EMD (`emd.sift.ensemble_sift`;
[@Wu2009]), Masked EMD (`emd.sift.mask_sift`; [@Deering2005]) and the
second-level sift (`emd.sift.sift_second_layer`; [@Huang2016] These high-level
functions rest upon a range of lower-level functions which are readily usable
by an advanced user. All levels of the sift computation are highly customisable
from the top-level sift functions. Users can configure these sift options using
a dictionary-like `emd.sift.SiftConfig` object. This config can then be passed
directly to the sift functions or saved in yaml format for later use or
sharing.

Each IMF can be analysed in terms of its instantaneous frequency
characteristics at the full temporal resolution of the dataset [@Huang2009].
The Hilbert-transform is used to construct an energy-frequency or
energy-frequency-time spectrum known as the Hilbert-Huang Transform (HHT). A
second level decomposition of the amplitude modulations of each IMF extends the
HHT to the Holospectrum describing signal energy across carrier frequency,
amplitude modulation frequency and time. The frequency transforms are
implemented in the `emd.spectra` submodule. `emd.spectra.frequency_stats`
implements a set of methods for computing instantaneous frequency, phase and
amplitude from a set of IMFs. These can be used as inputs to the
`emd.spectra.hilberthuang` or `emd.spectra.holospectrum` to energy
distributions across time and frequency. The Hilbert-Huang and Holospectrum
computations can be very large so these functions use an efficient sparse array
implementation.

The EMD toolbox provides a range of functions for the detection of oscillatory
cycles from the IMFs of a signal. Once identified, each cycle can be
characterised by a range of features including its amplitude, frequency and
waveform shape. Tools are provided for detecting continuous chains of
oscillatory cycles and for matching similar cycles across datasets. The cycle
analysis functions are implemented in `emd.cycle`

A range of utility and support features are included in the EMD toolbox.
Firstly, an easy to use and customisable logger (implemented in `emd.logger`)
is threaded throughout the toolbox to provide progress output about ongoing
computations, warnings and errors. The logger output may be augmented by the
user and any output can be directed to a specified log file in addition to the
console. Secondly, `EMD` is supported by a range of tests implmemented in the
`py.test` framework. These include both routine useage tests and tests ensuring
that the behaviour of the sift routines meet a set of pre-specified
requirements. Finally, `emd.support` contains a set of functions for running
tests and checking which versions of `EMD` are currently installed and whether
updates are available on [PyPI](https://pypi.org/project/emd/).

# Target Audience

Since its initial publication in 1998, the EMD approach has had a wide impact
across science and engineering, finding applications in turbulance, fluid
dynamics, geology, biophysics and neuroscience amongst many others. The EMD
toolbox will be of interest to scientists, engineers and applied mathematicians
looking to characterise complicated and dynamic signals in a high resolution.
This toolbox was developed for applications in electrophysiology and
neuroscience but care has been taken to ensure that the routines are generic
and applicable to any time-series.

# State of the field

The popularity of the EMD algorithm has lead to several existing
implementations. Here, we include an imcomplete list of these toolboxes. In
Python, there are two substantial EMD implementations available on the PyPI
server. [PyEMD](https://pyemd.readthedocs.io/en/latest/) and
[PyHHT](https://pyhht.readthedocs.io/en/latest/). Each of these packages
implements a family of sifting routines and frequency transforms. Another
implementation of EMD in Matlab and C is available from [Patrick
Flandarin](http://perso.ens-lyon.fr/patrick.flandrin/emd.html). This provides a
wide range of sift functions but limited frequency transform or spectrum
computations. Finally, the basic EMD algorithm and HHT is implemented in
versions of the [MatLab signal processing
toolbox](https://uk.mathworks.com/help/signal/ref/emd.html)

# Installation & Contribution

The EMD package is implemented in Python (>3.5). freely available under a GPL-3
license from PyPI.org. Users and developers can also install from source from
[gitlab](https://gitlab.com/emd-dev/emd). Our
[documenatation](https://emd.readthedocs.io) provides detailed instructions on
[installation](https://emd.readthedocs.io/en/latest/install.html) and a range
of practical
[tutorials](https://emd.readthedocs.io/en/latest/emd_tutorials/index.html).
Finally, users wishing to submit bug reports or merge-requests are able to do
so on our gitlab page following our [contribution
guidelines](https://emd.readthedocs.io/en/latest/contributing.html).


# Acknowledgements

We would like to extend our sincere thanks to Norden E. Huang, Wei-Kuang Liang,
Jia-Rong Yeh, Chi-Hung Juan, Vitor Lopes-dos-Santos and David Dupret for
fruitful and enjoyable discussions on EMD methododology. We would also like to
thank Vitor Lopes-dos-Santos, Jasper Hajonides van der Meulen and Irene
Echeverria-Altuna for their time, patience and feedback on early versions of
this toolbox.

# References

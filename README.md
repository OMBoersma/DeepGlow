# DeepGlow: Neural Network emulation of BOXFIT
<p align="center">
<img src="Logo_DeepGlow.png" width="250" height="325"/>
</p>
DeepGlow is a feed-forward neural network trained to emulate BOXFIT simulation data of gamma-ray burst (GRB) afterglows.

This package provides an easy interface to generate GRB spectra and light curves mimicking those generated through BOXFIT with high accuracy. Details are provided in a forthcoming paper and this repository also contains the code used to generate the training data and to train the neural networks.

## Installation

Installation is straightforward via pip:

`pip install deepglow`

DeepGlow specifically requires TensorFlow 2 and the importlib.resources package.

## Use

To generate light curves or spectra, create an instance of the `Emulator` class specifying the progenitor environment (either `ism` or `wind`):

```
from DeepGlow import Emulator

model = Emulator(simtype='ism')
```

The `flux` function of the `Emulator` class returns the flux values in mJy. It takes the three arguments `params`, `times` and `nu` corresponding to an array of the GRB afterglow parameters, observing times in seconds and observing frequencies in Hz. Each observing time value in the `times` array must correspond to an observing frequency value in the `nu` array. The afterglow parameters need to be specified in the following order in the `params` array:

- $z$ : redshift.
- $\log\_{10} d\_{L,28}$ : luminosity distance (log10 of 10^28 cm).
- $\log\_{10} E\_\mathrm{K,iso,53}$ : isotropic-equivalent energy (log10 of 10^53 erg). 
- $\log\_{10} n\_\mathrm{ref}$ : circumburst medium density (log10 of cm^-3).
- $\theta\_0$ : jet half-opening angle (rad).
- $\theta\_\mathrm{obs} / \theta\_0$ : off-axis observer angle as a fraction of the jet half-opening angle.
- $p$ : electron spectral index.
- $\bar{\epsilon}\_e \equiv \frac{p-2}{p-1} \epsilon\_e$ : energy fraction in accelerated electrons (in log10), with factor of $(p - 2) / (p - 1)$ absorbed.
- $\epsilon\_B$ (in log10) : energy fraction in magnetic field (in log10).
- $\xi\_N$ : fraction of electrons accelerated (in log10).

For example:

```
import numpy as np

observing_times=np.array([1e5,1e6,1e7])
observing frequencies = np.array([1e9,1e12,1e15])
GRB_params = np.array([0,-1,0,0,0.1,0.1,2.2,-2,-2,0])
flux_values = model.flux(params=GRB_params,times=observing_times,nu=observing_frequencies)
print(flux_values)
# [5.75068180e-01, 8.58790301e-01, 5.39014321e-05]
```

## Training data

The training data generated with BOXFIT will be provided through a Zenodo data package.

## References

- DeepGlow paper: _Forthcoming_ 
- BOXFIT: van Eerten, H., Horst, A. v. d., & MacFadyen, A. 2012, The Astrophysical Journal, 749, 44

 

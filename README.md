# DeepGlow: Neural Network emulation of BOXFIT
<p align="center">
<img src="Logo_DeepGlow.png" width="250" height="325"/>
</p>
DeepGlow is a feed-forward neural network trained to emulate BOXFIT simulation data of gamma-ray burst (GRB) afterglows. This package provides an easy interface to generate GRB afterglow spectra and light curves mimicking those generated through BOXFIT with high accuracy. Details are provided in a preprint paper ([arXiv:2212.10943](https://arxiv.org/abs/2212.10943)). This repository also contains the code used to generate the training data and to train the neural networks.

## Installation

Installation is straightforward via pip:

`pip install DeepGlow`

DeepGlow specifically requires `TensorFlow 2.x.x` and the `importlib.resources` package.

## Use

To generate light curves or spectra, create an instance of the `Emulator` class specifying the progenitor environment (either `ism` or `wind`):

```
from DeepGlow import Emulator

model = Emulator(simtype='ism')
```

The `flux` function of the `Emulator` class returns the flux values in mJy. It takes the three arguments `params`, `t_obs` and `nu_obs` corresponding to an array of the GRB afterglow parameters, observing times in seconds and observing frequencies in Hz. Each observing time value in the `t_obs` array must correspond to an observing frequency value in the `nu_obs` array. The afterglow parameters need to be specified in the following order in the `params` array:

- $z$ : redshift.
- $\log\_{10} d\_{L,28}$ : luminosity distance (log10 of 10^28 cm).
- $\log\_{10} E\_\mathrm{K,iso,53}$ : isotropic-equivalent energy (log10 of 10^53 erg). 
- $\log\_{10} n\_\mathrm{ref}$ : circumburst medium density (log10 of cm^-3).
- $\theta\_0$ : jet half-opening angle (rad).
- $\theta\_\mathrm{obs} / \theta\_0$ : off-axis observer angle as a fraction of the jet half-opening angle.
- $p$ : electron spectral index.
- $\log\_{10} \bar{\epsilon}\_e \equiv \frac{p-2}{p-1} \epsilon\_e$ : energy fraction in accelerated electrons (in log10), with factor of $(p - 2) / (p - 1)$ absorbed.
- $\log\_{10} \epsilon\_B$ : energy fraction in magnetic field (in log10).
- $\log\_{10} \xi\_N$ : fraction of electrons accelerated (in log10).

For example:

```
import numpy as np

GRB_params = np.array([0,-1,0,0,0.1,0.1,2.2,-2,-2,0])
observing_times=np.array([1e5,1e6,1e7])
observing_frequencies = np.array([1e9,1e12,1e15])
flux_values = model.flux(params=GRB_params,t_obs=observing_times,nu_obs=observing_frequencies)
print(flux_values)
# [5.75068180e-01, 8.58790301e-01, 5.39014321e-05]
```

## Training data

The training data generated with BOXFIT is provided through a Zenodo data package: https://zenodo.org/record/7472542

## References

- DeepGlow paper: https://arxiv.org/abs/2212.10943 
- BOXFIT: van Eerten, H., Horst, A. v. d., & MacFadyen, A. 2012, The Astrophysical Journal, 749, 44

 

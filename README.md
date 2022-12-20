# DeepGlow: Neural Network emulation of BOXFIT

DeepGlow is a feed-forward neural network trained to emulate BOXFIT simulation data of gamma-ray burst (GRB) afterglows.

This package provides an easy interface to generate GRB spectra and light curves mimicking those generated through BOXFIT with high accuracy. Details are provided in a forthcoming paper and this repository also contains the code used to generate the training data and to train the neural networks.

## Installation

Installation is straightforward via pip:

'pip install deepglow'

DeepGlow specifically requires TensorFlow 2 and the importlib.resources package.

## Use

To generate light curves or spectra, create an instance of the `Emulator` class specifying the progenitor environment (either `ism` or `wind`):

```
from DeepGlow import Emulator

model = Emulator(simtype='ism')
```

The `flux` function of the `Emulator` class returns the flux values in mJy. It takes the three arguments `params`, `times` and `nu` corresponding to an array of the GRB afterglow parameters, observing times in seconds and observing frequencies in Hz. The afterglow parameters must be specified as follows:

- 0: Redshift $z$.
- 1: Luminosity distance $\log10 d\_{L,28}$ (log10 of 10^28 cm).
- 2: Isotropic-equivalent energy (log10 of 10^53 erg) $\log10 E\_\mathrm{K,iso,53}$
- 3: Circumburst medium density (log10 of cm^-3)
- 4: Jet half-opening angle (rad)
- 5: Off-axis observer angle as a fraction of the jet half-opening angle
- 6: Electron spectral index
- 7: Energy fraction in accelerated electrons, with factor of `(p - 2) / (p - 1)` absorbed: `eps_e_bar = eps_e * (p - 2) / (p - 1)`
- 8: Energy fraction in magnetic field
- 9: Fraction of electrons accelerated.

## Training data

The training data generated with BOXFIT will be provided through a Zenodo data package.

## References

- DeepGlow paper: _Forthcoming_ 
- BOXFIT: van Eerten, H., Horst, A. v. d., & MacFadyen, A. 2012, The Astrophysical Journal, 749, 44

 

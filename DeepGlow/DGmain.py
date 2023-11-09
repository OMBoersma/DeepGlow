from typing import List
import numpy as np
from tensorflow import keras
import importlib.resources
import sys


class Emulator:
    """
    A class to emulate BOXFIT GRB afterglow physics using a pretrained neural network model.
    """

    def __init__(self, simtype: str = "ism") -> None:
        """
        Initializes the Emulator with the specified simulation type ('ism' or 'wind').

        Args:
            simtype (str): The type of simulation to perform. Defaults to 'ism'.
        """
        # Load scaling factors and model based on the simulation type
        self.load_resources(simtype)

        # Fixed model parameters
        self.Nparams = 8
        self.nDP = 117  # Number of data points
        self.sdays = 60 * 60 * 24  # Seconds in a day
        self.tcomp = np.geomspace(0.1, 1000, self.nDP) * self.sdays
        self.ref_d_L = 50 * 3.08567758 * 1e24  # Reference luminosity distance

    def load_resources(self, simtype: str) -> None:
        """
        Loads the scaling factors and neural network model from resources.

        Args:
            simtype (str): The type of simulation to perform.
        """
        filename = f"scale_facs_{simtype}_final.csv"
        modelname = f"model-{simtype}-final.hdf5"
        if sys.version_info >= (3, 9):
            # Use the newer `files` interface available in Python 3.9+
            with importlib.resources.files("DeepGlow").joinpath("data") as data_path:
                scale_path = data_path / filename
                model_path = data_path / modelname

                # Load scaling factors for standard scaling, 117 total datapoints
                scale_facs = np.loadtxt(scale_path.absolute().as_posix())
                self.Xmean = scale_facs[:-234][::2]
                self.Xstd = scale_facs[:-234][1::2]
                self.Ymean = scale_facs[-234:][::2]
                self.Ystd = scale_facs[-234:][1::2]

                # Load the model
                self.NNmodel = keras.models.load_model(
                    model_path.absolute().as_posix(), compile=False
                )
        else:
            # Fallback to the older `path` interface for Python versions < 3.9
            with importlib.resources.path("DeepGlow", "data") as data_path:
                scale_path = data_path / filename
                model_path = data_path / modelname

                # Load scaling factors for standard scaling, 117 total datapoints
                scale_facs = np.loadtxt(scale_path.absolute().as_posix())
                self.Xmean = scale_facs[:-234][::2]
                self.Xstd = scale_facs[:-234][1::2]
                self.Ymean = scale_facs[-234:][::2]
                self.Ystd = scale_facs[-234:][1::2]

                # Load the model
                self.NNmodel = keras.models.load_model(
                    model_path.absolute().as_posix(), compile=False
                )

    def flux(
        self, params: List[float], t_obs: np.ndarray, nu_obs: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the flux for given parameters, observation times, and frequencies.

        Args:
            params (list): The parameters for the flux calculation.
            t_obs (np.array): The observation times.
            nu_obs (np.array): The observation frequencies.

        Returns:
            np.array: The calculated flux for each observation time.
        """
        # Unpack parameters
        (
            z,
            log10_d_L_28_,
            log10_E_iso_53_,
            log10_n_ref_,
            theta_c,
            theta_jn,
            p,
            log10_eps_e_bar_,
            log10_eps_B_,
            log10_xi_N_,
        ) = params
        theta_jn *= theta_c  # Adjust jet angle based on core angle
        xi_N = 10**log10_xi_N_

        # Convert parameters to physical units
        E0 = (10**log10_E_iso_53_) * 1e53 * xi_N
        d_L = (10**log10_d_L_28_) * 1e28
        n0 = (10**log10_n_ref_) * xi_N
        ee = (10**log10_eps_e_bar_) * ((p - 1) / (p - 2)) * (1.0 / xi_N)
        eB = (10**log10_eps_B_) * (1.0 / xi_N)

        # Adjust times and frequencies for redshift
        t_obs = t_obs / (1 + z)
        nu_obs = nu_obs * (1 + z)

        # Ensure non-zero jet angle
        theta_jn = max(theta_jn, 1e-6)

        # Prepare input array for the neural network
        nu_unique = np.unique(nu_obs)
        nu_inds = [np.where(nu_obs == nu)[0] for nu in nu_unique]
        f_obs = np.zeros(len(t_obs))
        inp_arr = np.array(
            [
                [
                    np.log10(E0),
                    np.log10(theta_jn),
                    np.log10(theta_c),
                    np.log10(n0),
                    p,
                    np.log10(ee),
                    np.log10(eB),
                    np.log10(nu),
                ]
                for nu in nu_unique
            ]
        )

        # Predict the output from the neural network, calling NNmodel directly slightly faster than predict function.
        outY_unscaled = (
            10
            ** (
                (self.NNmodel((inp_arr - self.Xmean) / self.Xstd)) * self.Ystd
                + self.Ymean
            ).numpy()
        )

        # Interpolate the neural network output to the observation times
        for i, nu in enumerate(nu_unique):
            t_nu = t_obs[nu_inds[i]]
            dataOut = np.interp(t_nu, self.tcomp, outY_unscaled[i, :])
            f_obs[nu_inds[i]] = dataOut

        # Scale the flux to the observation
        f_obs = f_obs * (1.0 + z) / ((d_L / self.ref_d_L) ** 2)

        return f_obs

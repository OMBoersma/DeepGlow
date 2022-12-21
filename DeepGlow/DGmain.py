import numpy as np
from tensorflow import keras
import importlib.resources

class Emulator(object):

    def __init__(self, simtype='ism'):

        if simtype == 'ism':
            with importlib.resources.path('DeepGlow', 'data') as data_path:
                scale_path = data_path / "scale_facs_ism_final.csv"
                scale_txt = scale_path.absolute().as_posix()
                scale_facs = np.loadtxt(scale_txt)
                self.Xmean = scale_facs[:-234][::2]
                self.Xstd = scale_facs[:-234][1::2]
                self.Ymean = scale_facs[-234:][::2]
                self.Ystd = scale_facs[-234:][1::2]
                model_path = data_path / "model-ism-final.hdf5"
                model_hdf = model_path.absolute().as_posix()
                self.NNmodel = keras.models.load_model(model_hdf, compile=False)
        elif simtype == 'wind':
            with importlib.resources.path('DeepGlow', 'data') as data_path:
                scale_path = data_path / "scale_facs_wind_final.csv"
                scale_txt = scale_path.absolute().as_posix()
                scale_facs = np.loadtxt(scale_txt)
                self.Xmean = scale_facs[:-234][::2]
                self.Xstd = scale_facs[:-234][1::2]
                self.Ymean = scale_facs[-234:][::2]
                self.Ystd = scale_facs[-234:][1::2]
                model_path = data_path / "model-wind-final.hdf5"
                model_hdf = model_path.absolute().as_posix()
                self.NNmodel = keras.models.load_model(model_hdf, compile=False)
        # Fixed model parameters
        self.Nparams = 8
        self.nDP = 117
        self.sdays = 60*60*24
        self.tcomp = np.geomspace(0.1, 1000, self.nDP)*self.sdays
        self.ref_d_L = 50*3.08567758 * 1e24

    def flux(self, params, t_obs, nu_obs):
        z = params[0]
        log10_d_L_28_ = params[1]
        log10_E_iso_53_ = params[2]
        log10_n_ref_ = params[3]
        theta_c = params[4]
        theta_jn = params[5]*theta_c
        p = params[6]
        log10_eps_e_bar_ = params[7]
        log10_eps_B_ = params[8]
        log10_xi_N_ = params[9]
        xi_N = 10**log10_xi_N_
        E0 = (10**(log10_E_iso_53_)) * 1e53 * (xi_N)
        d_L = (10**(log10_d_L_28_)) * 1e28
        n0 = (10**(log10_n_ref_)) * (xi_N)
        ee = (10**(log10_eps_e_bar_)) * ((p-1)/(p-2)) * (1.0/xi_N)
        eB = (10**(log10_eps_B_)) * (1.0/xi_N)
        t_obs = t_obs / (1+z)
        nu_obs = nu_obs*(1+z)
        if theta_jn == 0:
            theta_jn = 1e-6
        nu_unique = np.unique(nu_obs)
        nu_inds = [np.where(nu_obs == nu)[0] for nu in nu_unique]
        f_obs = np.zeros(len(t_obs))
        inp_arr = np.zeros((len(nu_unique), self.Nparams))
        inp_arr[:, :] = [(np.log10(E0), np.log10(theta_jn), np.log10(theta_c), np.log10(
            n0), p, np.log10(ee), np.log10(eB), np.log10(nu)) for nu in nu_unique]
        outY_unscaled = 10**((self.NNmodel((inp_arr - self.Xmean) /
                             self.Xstd)) * self.Ystd + self.Ymean).numpy()
        for i, nu in enumerate(nu_unique):
            t_nu = t_obs[nu_inds[i]]
            dataOut = np.interp(t_nu, self.tcomp, outY_unscaled[i, :])
            f_obs[nu_inds[i]] = dataOut
        f_obs = f_obs*(1.0+z)/((d_L/self.ref_d_L)**2)
        return f_obs

from DeepGlow import Emulator
import numpy as np

model = Emulator(simtype='ism')

observing_times=np.array([1e5,1e6,1e7])
observing_frequencies = np.array([1e9,1e12,1e15])
GRB_params = np.array([0,-1,0,0,0.1,0.1,2.2,-2,-2,0])
flux_values = model.flux(params=GRB_params,t_obs=observing_times,nu_obs=observing_frequencies)
print(flux_values)

# [5.75068180e-01, 8.58790301e-01, 5.39014321e-05]

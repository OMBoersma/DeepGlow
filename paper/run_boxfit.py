import numpy as np
import time
import pandas as pd
import subprocess
import sys
from whichbox import which_boxes

n = int(2500)
Mpc_cm = 3.08567758 * 1e24
redshift = 0  
filenumber = sys.argv[1]
nDP = 117
dp = []
for k in range(nDP):
    dp.append('dp'+str(k+1))

data = pd.DataFrame(index=np.arange(int(n)), columns=[
                    'Eiso', 'theta_jn', 'theta_c', 'n0', 'p', 'epsilon_e', 'epsilon_B', 'nu']+dp)
sdays = 60*60*24
tdata = np.geomspace(0.1, 1000, nDP)*sdays
E = np.random.uniform(50, 56, n)
theta_c = np.random.uniform(np.log10(0.01), np.log10(0.5*np.pi), n)
theta_obs = np.zeros(n)
for i, th in enumerate(theta_c):
    if 2*(10**th) <0.5*np.pi:
    	theta_obs[i] = np.random.uniform(0.01, 2*(10**th))
    else:
	theta_obs[i] = np.random.uniform(0.01, 0.5*np.pi)
n0 = np.random.uniform(-5, 3, n)
p = np.random.uniform(2, 3, n)
ee = np.random.uniform(-10, 0, n)
eB = np.random.uniform(-10, 0, n)
nu_obs = np.random.uniform(8, 19, n)
data.loc[:, :-nDP] = np.c_[E, theta_obs, theta_c, n0, p, ee, eB, nu_obs]
root = '/home/oboersma/boxfit/lightcurve_dataset/data'
boxroot = '/home/oboersma/boxfit/bin'
times = []
for i in range(n):
    E_i, theta_obs_i, theta_c_i, n0_i, p_i, ee_i, eB_i, nu_i = data.iloc[i, :-nDP]
    E_i = 10**E_i
    theta_c_i = 10**theta_c_i
    n0_i = 10**n0_i
    ee_i = 10**ee_i
    eB_i = 10**eB_i
    nu_i = 10**nu_i
    boxes = which_boxes(theta_c_i)
    t0 = time.time()
    cmd = "mpirun -n 40 --use-hwthread-cpus {boxroot}/boxfit_noboost -t_0={t0} -t_1={t1} -nu_0={nu} -nu_1={nu} -d_L={d_L} -z={z} -theta_0={theta_0} -E={E0} -n={n0} -theta_obs={theta_obs} -p={p} -epsilon_B={eB} -epsilon_E={eE} -ksi_N={ksi_N} -no_points={no_points} -box0={box0} -box1={box1}".format(
        boxroot=boxroot, t0=tdata[0], t1=tdata[-1], nu=nu_i, d_L=50*Mpc_cm, z=redshift, theta_0=theta_c_i, E0=E_i, n0=n0_i, theta_obs=theta_obs_i, p=p_i, eB=eB_i, eE=ee_i, ksi_N=1, no_points=int(nDP), box0=boxes[0], box1=boxes[1])
    p = subprocess.run(cmd, shell=True, capture_output=True).stdout
    t1 = time.time()
    try:
        lines = p.splitlines()[-int(nDP+1):-1]
        lc_data = [float(str(line).split(",")[-1][1:-1]) for line in lines]
        data.loc[i, -nDP:] = np.log10(lc_data)
        times.append(t1-t0)
    except:
        print(p,flush=True)
    if not i % 100:
        print(i,flush=True)
np.savetxt(root+'/timing/time_'+filenumber+'.txt', times)
data.to_csv(root+'/lcdata_'+filenumber+'.csv', index=False)

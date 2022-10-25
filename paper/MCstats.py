import pickle
import sys
import numpy as np
import json
MCdist_file_scalefit_ism = open('scalefit-final-ism.pickle','rb')
MCdist_file_scalefit_wind = open('scalefit-final-wind.pickle','rb')
MCdist_file_deepglow_ism = open('deepglow-final-ism.pickle','rb')
MCdist_file_deepglow_wind = open('deepglow-final-wind.pickle','rb')

json_stats_scalefit_ism = json.load(open('scalefit-final-ism-stats.json','r'))
json_stats_deepglow_ism = json.load(open('deepglow-final-ism-stats.json','r'))
json_stats_scalefit_wind = json.load(open('scalefit-final-wind-stats.json','r'))
json_stats_deepglow_wind = json.load(open('deepglow-final-wind-stats.json','r'))

lnZ_scalefit_ism =json_stats_scalefit_ism['global evidence']
lnZ_scalefit_wind =json_stats_scalefit_wind['global evidence']
lnZ_deepglow_ism =json_stats_deepglow_ism['global evidence']
lnZ_deepglow_wind =json_stats_deepglow_wind['global evidence']

e = np.e
print('Bayes factor Z_wind_scalefit/Z_ism_scalefit : '+str( (e**lnZ_scalefit_wind) / (e**lnZ_scalefit_ism)))
print('Bayes factor Z_wind_deepglow/Z_ism_deepglow : '+str( (e**lnZ_deepglow_wind) / (e**lnZ_deepglow_ism)))
print('Bayes factor Z_ism_deepglow/Z_ism_scalefit : '+str( (e**lnZ_deepglow_ism) / (e**lnZ_scalefit_ism)))
print('Bayes factor Z_wind_deepglow/Z_wind_scalefit : '+str( (e**lnZ_deepglow_wind) / (e**lnZ_scalefit_wind)))

cache_scalefit_ism = pickle.load(MCdist_file_scalefit_ism)
cache_scalefit_wind = pickle.load(MCdist_file_scalefit_wind)
cache_deepglow_ism = pickle.load(MCdist_file_deepglow_ism)
cache_deepglow_wind = pickle.load(MCdist_file_deepglow_wind)

caches = [cache_scalefit_ism,cache_scalefit_wind,cache_deepglow_ism,cache_deepglow_wind]
names = ['scalefit_ism','scalefit_wind','deepglow_ism','deepglow_wind']
prmnames_latex = [r'$\log_{10}\theta_0$',r'$\log_{10}E_{K, \mathrm{iso}}$', r'$\log_{10}n_{\mathrm{ref}}$',r'$\theta_{\mathrm{obs}} / \theta_0$',r'$p$',r'$\log_{10}\epsilon_B$',r'$\log_{10}\bar{\epsilon}_e$',r'$A_V$']
paramnames=['theta_0','E_K_iso','n_ref','theta_obs_frac','p','epsilon_B','epsilon_e_bar','A_V']


class Param():
    def __init__(self):
        self.median = None
        self.upper = None
        self.lower = None
        self.samples = None

dict_list = []
for i,dist in enumerate(caches):
    marge_stat = dist.getMargeStats()
    prms = dist.getParams()
    prm_dict = {}
    for param_name in paramnames:
        density = dist.get1DDensity(param_name)
        samples = getattr(prms,param_name)
        _min, _max, _has_min, _has_top = density.getLimits([0.6827])
        median = np.median(samples)
        lower = _min - median 
        upper = _max - median
        prm_class = Param()
        prm_class.lower = lower
        prm_class.median = median
        prm_class.upper = upper
        prm_class.samples = samples
        prm_dict[param_name] =prm_class
    dict_list.append(prm_dict)
sf_ism_dict = dict_list[0]
sf_wind_dict = dict_list[1]
dg_ism_dict = dict_list[2]
dg_wind_dict = dict_list[3]

ebar_sf_wind = 10**sf_wind_dict['epsilon_e_bar'].samples 
p_sf_wind = sf_wind_dict['p'].samples
ebar_dg_wind =10**dg_wind_dict['epsilon_e_bar'].samples
p_dg_wind = dg_wind_dict['p'].samples
ee_sf_wind = ebar_sf_wind * ((p_sf_wind - 1) / (p_sf_wind - 2))
ee_dg_wind = ebar_dg_wind * ((p_dg_wind - 1) / (p_dg_wind - 2))
print(np.median(ee_sf_wind))
with open('combined_table_ism.txt','w') as table:
    table.write('\\begin{table}\n')
    table.write('\\begin{tabular} {| l | l | l | l |}\n')
    table.write('\\hline\n')
    table.write('Parameter & \\texttt{DeepGlow} & \\texttt{ScaleFit} & Match \\\\\n')
    table.write('\\hline\n')
    for j,param_name in enumerate(paramnames):
        prm_dg_ism = dg_ism_dict[param_name]
        prm_sf_ism = sf_ism_dict[param_name]
        if (prm_dg_ism.median + prm_dg_ism.upper > prm_sf_ism.median + prm_sf_ism.lower and prm_dg_ism.median < prm_sf_ism.median) or (prm_dg_ism.median + prm_dg_ism.lower < prm_sf_ism.median + prm_sf_ism.upper and prm_dg_ism.median > prm_sf_ism.median):
            line = r'{'+prmnames_latex[j]+r'}' + r' & $'+ '%.2f'%(prm_dg_ism.median) +r'^{+' + '%.2f'%(prm_dg_ism.upper)+r'}_{'+'%.2f'%(prm_dg_ism.lower)+r'}$' + r' & $'+ '%.2f'%(prm_sf_ism.median) +r'^{+' + '%.2f'%(prm_sf_ism.upper)+r'}_{'+'%.2f'%(prm_sf_ism.lower)+r'}$ & $ \quad \checkmark$ \\'+'\n'
        else:
            line = r'{'+prmnames_latex[j]+r'}' + r' & $'+ '%.2f'%(prm_dg_ism.median) +r'^{+' + '%.2f'%(prm_dg_ism.upper)+r'}_{'+'%.2f'%(prm_dg_ism.lower)+r'}$' + r' & $'+ '%.2f'%(prm_sf_ism.median) +r'^{+' + '%.2f'%(prm_sf_ism.upper)+r'}_{'+'%.2f'%(prm_sf_ism.lower)+r'}$ & $ \quad \times$ \\'+'\n'            
        table.write(line)
    table.write('\\hline\n')
    table.write('\\end{tabular}\n')
    table.write('\\caption{ISM}')
    table.write('\n\\end{table}')
with open('combined_table_wind.txt','w') as table:
    table.write('\\begin{table}\n')
    table.write('\\begin{tabular} {| l | l | l | l |}\n')
    table.write('\\hline\n')
    table.write('Parameter & \\texttt{DeepGlow} & \\texttt{ScaleFit} & Match \\\\\n')
    table.write('\\hline\n')
    for j,param_name in enumerate(paramnames):
        prm_dg_wind = dg_wind_dict[param_name]
        prm_sf_wind = sf_wind_dict[param_name]
        if (prm_dg_wind.median + prm_dg_wind.upper > prm_sf_wind.median + prm_sf_wind.lower and prm_dg_wind.median < prm_sf_wind.median) or (prm_dg_wind.median + prm_dg_wind.lower < prm_sf_wind.median + prm_sf_wind.upper and prm_dg_wind.median > prm_sf_wind.median):
            line = r'{'+prmnames_latex[j]+r'}' + r' & $'+ '%.2f'%(prm_dg_wind.median) +r'^{+' + '%.2f'%(prm_dg_wind.upper)+r'}_{'+'%.2f'%(prm_dg_wind.lower)+r'}$' + r' & $'+ '%.2f'%(prm_sf_wind.median) +r'^{+' + '%.2f'%(prm_sf_wind.upper)+r'}_{'+'%.2f'%(prm_sf_wind.lower)+r'}$ & $ \quad \checkmark$ \\'+'\n'
        else:
            line = r'{'+prmnames_latex[j]+r'}' + r' & $'+ '%.2f'%(prm_dg_wind.median) +r'^{+' + '%.2f'%(prm_dg_wind.upper)+r'}_{'+'%.2f'%(prm_dg_wind.lower)+r'}$' + r' & $'+ '%.2f'%(prm_sf_wind.median) +r'^{+' + '%.2f'%(prm_sf_wind.upper)+r'}_{'+'%.2f'%(prm_sf_wind.lower)+r'}$ & $ \quad \times$ \\'+'\n'
        table.write(line)
    table.write('\\hline\n')
    table.write('\\end{tabular}\n')
    table.write('\\caption{wind}')
    table.write('\n\\end{table}')



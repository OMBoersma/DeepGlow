import pickle
from getdist import plots
from matplotlib import pyplot as plt
import sys
from matplotlib import rc

f1 = sys.argv[1]
f2 = sys.argv[2]
f3 = sys.argv[3]

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

with open(f1,mode='rb') as f:
    scalefit = pickle.load(f)
with open(f2,mode='rb') as f:
    deepglow = pickle.load(f)

defaultSettings = plots.GetDistPlotSettings()
defaultSettings.rcSizes(axes_fontsize=12, lab_fontsize=12)
g = plots.getSubplotPlotter(width_inch=10)
g.settings.rcSizes(axes_fontsize=12,lab_fontsize=15)
g.settings.num_plot_contours = 5
g.settings.legend_fontsize = 20
g.triangle_plot([scalefit,deepglow],legend_labels=[r'\texttt{ScaleFit}',r'\texttt{DeepGlow}'],params=['theta_0','E_K_iso','n_ref','theta_obs_frac','p','epsilon_B','epsilon_e_bar','A_V'],filled=True, colors='Set1')
plt.savefig(f3+'.png', dpi=400)

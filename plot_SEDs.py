'''
this script plots the SEDs of both NGC2071 and IRAS20050+2720
'''
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import photometry as p
import astropy.constants as const
from astropy.table import Table
import seaborn as sns
from plot_library import SED_mosaic

folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"



# load plot params
sns.set(context='paper',style='ticks')
mpl.rcParams['xtick.labelsize']=14
mpl.rcParams['ytick.labelsize']=14
mpl.rcParams['axes.labelsize']=16
mpl.rcParams['legend.fontsize']=14
mpl.rcParams['font.size']=14

### FIGURE WITH MOSAIC OF SEDs ###
fig2071 = plt.figure(figsize=(15,15))
SED_mosaic(cluster='NGC2071',
	fig_per_line = 3,
	margin=None,
	left_offset=0.1,
	right_offset=0.,
	top_offset=0.06,
	bottom_offset=0.07,
	show=False,
	figure=fig2071)

figIRAS = plt.figure(figsize=(15,15))
SED_mosaic(cluster='IRAS20050',
	fig_per_line = 3,
	margin=None,
	left_offset=0.1,
	right_offset=0.,
	top_offset=0.06,
	bottom_offset=0.07,
	show=False,
	figure=figIRAS)
	
	

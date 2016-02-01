'''
this script creates images for the IRAS20050 & NGC2071 paper
'''

import numpy as np
import aplpy
from astropy.coordinates import SkyCoord
import pickle
from aplpy import wcs_util
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from plot_library import plot_one,plot_img_mosaic,plot_RGB
import matplotlib.gridspec as gridspec

folder_ngc2071 = "/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/Spitzer-2071/"
folder_herschel = "/cardini3/mrizzo/2012SOFIA/Herschel_Mosaics/Herschel-2071/"
folder_mosaics = "/n/a2/mrizzo/Dropbox/SOFIA/Mosaics/"
folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"

########################################################################################################
## IRAS20050+2720 PLOT I1 to F37 ##
#figIRAS20050 = plt.figure(figsize=(15,10))
#plot_img_mosaic(figure=figIRAS20050)



########################################################################################################
## NGC2071 PLOT Spitzer RGB image of whole region ##
fig = plt.figure(figsize=(15,7.5))
gs = gridspec.GridSpec(1,2)
gs.update(wspace=0.2,hspace=0.0)

R = "/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/Spitzer-2071/NGC2071-short.IRAC.4.mosaic.fits"
G = "/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/Spitzer-2071/NGC2071-short.IRAC.3.mosaic.fits"
B = "/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/Spitzer-2071/NGC2071-short.IRAC.1.mosaic.fits"

name = 'NGC2071_RGB'
ax = plt.subplot(gs[0])
ax.set_xticks([])
ax.set_yticks([])
pos = ax.get_position()
print [pos.x0,pos.y0,pos.width,pos.height]

plot_RGB(
	figname = fig,
	subplot=[pos.x0,pos.y0,pos.width,pos.height],
	filename = [R,G,B],
	title = name,
	rgb = [3,99.9],rgb_stretch=['linear'], # can put lists here
	show_circles = False,show_grid=True,show_xaxis=True,show_yaxis=True,show_contours=True,
	show_scalebar=True,show=False,show_labels=False,
	show_focus=False,focus = [],
	labels=[],
	show_fields=True, fields=[['5:47:07.377','+0:21:38.70',-6,180],['5:47:07.479','+0:18:13.65',-4,180]],
	scale_bar=60, #arcsec
	dist = 390., # pc
	name='NGC2071', contours=None,
	contours_levels=[],
	radius_source = 2.4, # arcsec
	ra='5:47:07.371',dec='+0:20:00.86',
	radius = 250.) # of snapshot, arcsec

R = "/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/IRAS20050short.IRAC.4.mosaic.fits"
G = "/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/IRAS20050-short.IRAC.3.mosaic.fits"
B = "/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/IRAS20050short.IRAC.1.mosaic.fits"
name = 'IRAS20050_RGB'
ax = plt.subplot(gs[1])
ax.set_xticks([])
ax.set_yticks([])
pos = ax.get_position()

plot_RGB(figname = fig,filename = [R,G,B],title = name,show_grid=True,rgb_stretch=['linear','linear','linear'],subplot=[pos.x0,pos.y0,pos.width,pos.height],show=True)

fig.savefig(folder_export+'RGB.png',dpi=300)

########################################################################################################
## NGC2071 Mosaic plot ##

fig2071_mosaic = plt.figure(figsize=(15,10))
list_fig = [folder_ngc2071+ 'NGC2071-short.IRAC.1.mosaic.fits',folder_ngc2071+ 'NGC2071-short.IRAC.4.mosaic.fits',
	folder_mosaics + 'NGC2071.11.fits',folder_mosaics + 'NGC2071.19.fits',
#	folder_ngc2071+ 'NGC2071.MIPS.1.mosaic.fits',
	folder_mosaics + 'NGC2071.31.fits',	folder_mosaics + 'NGC2071.37.fits',
#	folder_herschel + 'n2071-70um.fits']
	]
plot_img_mosaic(
	show=False,
	list_fig = list_fig,
	vmax=[500.,3500.,1.,3.,5.,8.],	
	vmin=[0.,0.,0.,0.,0.,0.],	
	list_titles = ['I1','I4','F11','F19','F31','F37'],
	beams = [2.,2.,3.,3.,3.,3.], #arcsecs
	fig_per_line = 3,
	name = 'NGC2071_mosaic',
	ra='5:47:04.909',dec='+0:21:53.46',radius=23.,
	figure=fig2071_mosaic,
	margin=None)


fig2071_saturated_mosaic = plt.figure(figsize=(15,5))
list_fig = [folder_ngc2071+ 'NGC2071-short.IRAC.1.mosaic.fits',
	folder_ngc2071+ 'NGC2071.MIPS.1.mosaic.fits',
	folder_herschel + 'n2071-70um.fits']
	
plot_img_mosaic(
	show=False,
	list_fig = list_fig,
	vmax=[500.,500.,50.],	
	vmin=[0.,0.,0.,],	
	list_titles = ['I1','M24','PACS70'],
	beams = [2.,6.,6], #arcsecs
	fig_per_line = 3,
	name = 'NGC2071_saturated_mosaic',
	ra='5:47:04.909',dec='+0:21:53.46',radius=40.,
	figure=fig2071_saturated_mosaic,
	margin=None)


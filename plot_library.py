'''
this script creates images for the IRAS20050 & NGC2071 paper
'''

import numpy as np
import aplpy
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import pickle
from aplpy import wcs_util
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
sns.set_style('ticks')
from photometry import markerPlotSEDax
import matplotlib.gridspec as gridspec

metafolder = "/cardini3/mrizzo/2012SOFIA/2014meta/"
folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"
folder_mosaics = "/n/a2/mrizzo/Dropbox/SOFIA/Mosaics/"
folder_spitzer = "/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/"
folder_ngc2071 = "/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/Spitzer-2071"

def plot_one(figname, subplot,filename,title,
		rgb = [],rgb_stretch=['linear'], # can put lists here
		show_circles = False,show_grid=False,beam=3,vmax=None,vmin=None,show_xaxis=True,show_yaxis=True,show_labels=False,
		dist = 700., # pc
		name='IRAS20050', contours=folder_mosaics + 'IRAS20050.37.fits',
		contours_levels=[0.0326531,0.0734694,0.130612,0.204082,0.293878,0.4],
		radius_source = 2.4, # arcsec
		ra='20h07m06.70s',dec='27d28m54.5s',
		radius = 17.): # of snapshot, arcsec
	'''
	this function displays the file in filename in all a uniform format
	all defaults to IRAS20050
	dist in pc
	radius in asec
	radius_source in asec
	'''
			
	# plot the mosaic FITS file
	fig = aplpy.FITSFigure(filename,figure=figname,subplot=subplot)	

	# coordinates of the center of the image
	c = SkyCoord(ra=ra,dec=dec,frame='fk5',unit=(u.hour,u.deg))

	# recenter image
	fig.recenter(c.ra.deg,c.dec.deg,radius=radius/3600.)

	# display image
	if isinstance(vmax,float):
		fig.show_colorscale(cmap='gist_earth',vmin=vmin,vmax=vmax)
	else:
		fig.show_colorscale(cmap='gist_earth')

	# load source list
	sources = pickle.load(open(folder_export+"totsourcetable_fits.data","r"))

	# extract only the RA, DEC column for name
	if show_circles:
		for i in range(len(sources)):
			if name in sources['SOFIA_name'][i] and "Total_Cluster" not in sources['Property'][i]:
				fig.show_circles(sources['RA'][i],sources['DEC'][i],radius_source/3600.,edgecolor='red',facecolor='none',alpha=0.8,lw=2)

	if show_labels:
		c = SkyCoord(ra='20h07m06.782s',dec='27d28m43.00s',frame='fk5')
		fig.add_label(c.ra.deg,c.dec.deg,'F1',color='red',weight='bold')
		c = SkyCoord(ra='20h07m06.798s',dec='27d28m56.48s',frame='fk5')
		fig.add_label(c.ra.deg,c.dec.deg,'F2',color='red',weight='bold')
		c = SkyCoord(ra='20h07m06.458s',dec='27d29m00.78s',frame='fk5')
		fig.add_label(c.ra.deg,c.dec.deg,'F3',color='red',weight='bold')
		c = SkyCoord(ra='20h07m05.592s',dec='27d28m59.61s',frame='fk5')
		fig.add_label(c.ra.deg,c.dec.deg,'F4',color='red',weight='bold')
		c = SkyCoord(ra='20h07m05.785s',dec='27d28m46.82s',frame='fk5')
		fig.add_label(c.ra.deg,c.dec.deg,'F5',color='red',weight='bold')
		
	# put axes on top
	fig.tick_labels.set_xposition('top')
	fig.axis_labels.set_xposition('top')
	fig.tick_labels.set_font(size='small')
	fig.tick_labels.set_xformat('hh:mm:ss')
	fig.tick_labels.set_yformat('dd:mm:ss')

	# turn grid on
	if show_grid:
		fig.add_grid()

	# show/hide axes
	if not show_xaxis:
		fig.axis_labels.hide_x()	
		fig.tick_labels.hide_x()
	if not show_yaxis:
		fig.axis_labels.hide_y()	
		fig.tick_labels.hide_y()

	
	# add title
	fig.add_label(0.2,0.9,title,relative=True,color='white',size=20)
	#fig.set_title(title)

	# add scalebar
	scale_bar = 5.
	scale_pc = 1./dist*scale_bar
	fig.add_scalebar(scale_bar/3600.)
	#fig.scalebar.set_frame(True)
	fig.scalebar.set_alpha(1.0)
	fig.scalebar.set_color('white')
	fig.scalebar.set_label('%d" = %.3f pc' % (scale_bar,scale_pc))
	fig.scalebar.set_linewidth(2)
#	fig.scalebar.set_font(weight='bold')
	fig.scalebar.set_font_size(15)


	# add beam indicator
	fig.add_beam(2.*beam/3600.,2.*beam/3600.,0.)
	fig.beam.set_color('white')
	fig.beam.set_alpha(0.8)
	fig.beam.set_linestyle('dashed')
	fig.beam.set_linewidth(2)

			
	# display contours
	if contours != None:
		fig.show_contour(contours,colors='white',returnlevels=True,levels=contours_levels)



def plot_img_mosaic(
	show=False,
	list_fig = [folder_spitzer+ 'IRAS20050short.IRAC.1.mosaic.fits',folder_spitzer+ 'IRAS20050short.IRAC.4.mosaic.fits',\
		folder_mosaics + 'IRAS20050.11.fits',folder_mosaics + 'IRAS20050.19.fits',folder_mosaics + 'IRAS20050.31.fits',\
		folder_mosaics + 'IRAS20050.37.fits'],
	vmax=[2500.,3500.,0.05,0.1,0.2,0.3],	
	vmin=[0.,0.,0.,0.,0.,0.],	
	list_titles = ['I1','I4','F11','F19','F31','F37'],
	beams = [2.,2.,2.,2.,3.,3.,3.,3.], #arcsecs
	fig_per_line = 3,
	margin=None,
	left_offset=0.1,
	right_offset=0.,
	top_offset=0.06,
	bottom_offset=0.07,
	name = 'IRAS20050',
	ra='20h07m06.70s',dec='27d28m54.5s',
	radius = 17.,
	show_circles=False,
	figure=None):
	'''
	PICTURE SHOWING ALL BANDS FROM I1 to F37
	'''
	# initiate figure
	if figure==None:
		fig = plt.figure(figsize=(15,15))
	else:
		fig=figure
	
	# number of figures
	N = len(list_fig)
	
	# number of rows
	num_rows = int(np.ceil(N/float(fig_per_line)))
	
	gs = gridspec.GridSpec(num_rows,fig_per_line)
	if margin != None:
		gs.update(left=left_offset,right=1.0-right_offset,top=1.0-top_offset,bottom=bottom_offset,wspace=margin,hspace=margin)

	# loop on the images
	for i in range(N):
		# only plot axes for left column and bottom row
		showx=showy=False
		if i % fig_per_line ==0:
			showy = True
		if i < fig_per_line:	
			showx = True
		print 'Plotting Figure ',i
		ax = plt.subplot(gs[i])
		ax.set_xticks([])
		ax.set_yticks([])
		pos = ax.get_position()
		plot_one(figname = fig, subplot=[pos.x0,pos.y0,pos.width,pos.height],filename=list_fig[i],vmax=vmax[i],vmin=vmin[i],\
			show_xaxis=showx,show_yaxis=showy,title = list_titles[i],beam=beams[i],ra=ra,dec=dec,radius=radius,show_circles=show_circles)
	if show:
		fig.show()
	fig.savefig(folder_export+name+'.png',dpi=300)

def plot_RGB(
	figname,
	filename,
	title,
	subplot=None,
	rgb = [10,99.75],rgb_stretch=['linear'], # can put lists here
	show_circles = False,show_grid=False,show_xaxis=True,show_yaxis=True,show_contours=True,
	show_scalebar=True,show=True,show_fields=True,show_labels=False,show_focus=False,
	focus = [], #ra,dec,size
	labels=[],
	fields=[['20h07m07.178s','+27d28m23.79s',-15,180],['20h07m02.561s','+27d30m25.90s',25,180]],
	scale_bar=20, #arcsec
	dist = 700., # pc
	name='IRAS20050', contours=None,
	contours_levels=[],
	radius_source = 2.4, # arcsec
	ra='20:07:06.70',dec='27:28:54.5',
	radius = 200.): # of snapshot, arcsec
	'''
	this function displays the file in filename in all a uniform format
	all defaults to IRAS20050
	dist in pc
	radius in asec
	radius_source in asec
	'''

	aplpy.make_rgb_cube([filename[0],filename[1],filename[2]],title+'_rgb_cube.fits')
	if len(rgb_stretch)==1:
		stretch_r= stretch_g= stretch_b = rgb_stretch[0]
	else:
		[stretch_r, stretch_g, stretch_b] = rgb_stretch
	if len(rgb)==2: 
		pmin_r=pmin_g=pmin_b=rgb[0]; pmax_r=pmax_g=pmax_b=rgb[1]
	else:
		[pmin_r,pmin_g,pmin_b,pmax_r,pmax_g,pmax_b] = rgb
	aplpy.make_rgb_image(title+'_rgb_cube.fits', title+'_rgb_cube.png',pmin_r=pmin_r,pmin_g=pmin_g,pmin_b=pmin_b,
		pmax_r=pmax_r,pmax_g=pmax_g,pmax_b=pmax_b,stretch_r=stretch_r, stretch_g=stretch_g, stretch_b=stretch_b)
	if subplot==None:
		fig = aplpy.FITSFigure(title+'_rgb_cube_2d.fits',figure=figname)
	else:
		fig = aplpy.FITSFigure(title+'_rgb_cube_2d.fits',figure=figname,subplot=subplot)
	fig.show_rgb(title+'_rgb_cube.png')	

	# coordinates of the center of the image
	c = SkyCoord(ra=ra,dec=dec,frame='fk5',unit=(u.hour,u.deg))

	# recenter image
	fig.recenter(c.ra.deg,c.dec.deg,radius=radius/3600.)
	
	# put axes on top
	fig.tick_labels.set_xposition('top')
	fig.axis_labels.set_xposition('top')
	fig.tick_labels.set_font(size='small')
	fig.tick_labels.set_xformat('hh:mm:ss')
	fig.tick_labels.set_yformat('dd:mm:ss')
	#fig.ticks.set_xspacing(10./3600.)

	# turn grid on
	if show_grid:
		fig.add_grid()

	# show/hide axes
	if not show_xaxis:
		fig.axis_labels.hide_x()	
		fig.tick_labels.hide_x()
	if not show_yaxis:
		fig.axis_labels.hide_y()	
		fig.tick_labels.hide_y()

	if show_labels:
		c = SkyCoord(ra='20h07m06.782s',dec='27d28m43.00s',frame='fk5',unit=(u.hour,u.deg))
		fig.add_label(c.ra.deg,c.dec.deg,'SOF1',color='red',weight='bold',size=20)
		c = SkyCoord(ra='20h07m06.798s',dec='27d28m56.48s',frame='fk5',unit=(u.hour,u.deg))
		fig.add_label(c.ra.deg,c.dec.deg,'SOF2',color='red',weight='bold',size=20)
		c = SkyCoord(ra='20h07m06.458s',dec='27d29m00.78s',frame='fk5',unit=(u.hour,u.deg))
		fig.add_label(c.ra.deg,c.dec.deg,'SOF3',color='red',weight='bold',size=20)
		c = SkyCoord(ra='20h07m05.592s',dec='27d29m03.0s',frame='fk5',unit=(u.hour,u.deg))
		fig.add_label(c.ra.deg,c.dec.deg,'SOF4',color='red',weight='bold',size=20)
		c = SkyCoord(ra='20h07m05.795s',dec='27d28m46.82s',frame='fk5',unit=(u.hour,u.deg))
		fig.add_label(c.ra.deg,c.dec.deg,'SOF5',color='red',weight='bold',size=20)

	# add title
	fig.add_label(0.2,0.9,title,relative=True,color='red',weight='bold')
	#fig.set_title(title)
	
	# load source list
	sources = pickle.load(open(folder_export+"totsourcetable_fits.data","r"))

	# extract only the RA, DEC column for name
	if show_circles:
		for i in range(len(sources)):
			if name in sources['SOFIA_name'][i] and "Total_Cluster" not in sources['Property'][i]:
				s = "%d" % (i)
				fig.show_circles(sources['RA'][i],sources['DEC'][i],radius_source/3600.,edgecolor='red',facecolor='none',alpha=0.8,lw=2,label=s)

	# add scalebar
	if show_scalebar:
		scale_pc = 1./dist*scale_bar
		fig.add_scalebar(scale_bar/3600.)
		#fig.scalebar.set_frame(True)
		fig.scalebar.set_alpha(0.7)
		fig.scalebar.set_color('red')
		fig.scalebar.set_label('%d" = %.3f pc' % (scale_bar,scale_pc))
		fig.scalebar.set_linewidth(3)
		fig.scalebar.set_font(weight='bold')

	# add another dashed rectangle showing the region of interest
	if show_focus:
		rafoc,decfoc,sizefoc=focus
		foc = SkyCoord(ra=rafoc,dec=decfoc,frame='fk5',unit=(u.hour,u.deg))
		fig.show_rectangles(foc.ra.deg,foc.dec.deg,sizefoc/3600.,decfoc/3600.,edgecolor='red',facecolor='none',alpha=0.8,lw=2,linestyle='dashed')
				
	# show the SOFIA fields
	if show_fields:
		patches = []
		for field in fields:
			raf,decf,ang,width = field
			c = SkyCoord(ra=raf,dec=decf,frame='fk5',unit=(u.hour,u.deg))

			xp, yp = wcs_util.world2pix(fig._wcs, c.ra, c.dec)
			wp = hp = width/3600. / wcs_util.celestial_pixel_scale(fig._wcs)
			
			rect = Rectangle((-wp/2., -hp/2), width=wp, height=hp)
			t1 = mpl.transforms.Affine2D().rotate_deg(ang).translate(xp, yp)
			rect.set_transform(t1)
			patches.append(rect)

		# add all patches to the collection
		p = PatchCollection(patches, edgecolor='white',facecolor='none',alpha=0.8,lw=2)
		
		# add collection to figure
		t = fig._ax1.add_collection(p)

	# display contours
	if contours != None:
		fig.show_contour(contours,colors='white',returnlevels=True,levels=contours_levels)

	if show:
		plt.show()


### FIGURE WITH MOSAIC OF SEDs ###

# number of figures
def SED_mosaic(cluster='NGC2071',
	fig_per_line = 3,
	margin=None,
	left_offset=0.1,
	right_offset=0.,
	top_offset=0.06,
	bottom_offset=0.07,
	show=False,
	figure=None):

	# load source table
	sourcetable = pickle.load(open(folder_export+"totsourcetable_fits.data","r"))
	sources = sourcetable.group_by('SOFIA_name')
	N=0
	for key,clustertable in zip(sources.groups.keys,sources.groups):
		if clustertable['Cluster'][0] == cluster and  (clustertable['Property'][0] == 'Isolated' or clustertable['Property'][0] == 'Clustered'):
			N+=1
	print N
	# number of rows
	num_rows = int(np.ceil(N/float(fig_per_line)))

	# size of a picture
#	size_v = 1./float(num_rows)*(1.0-margin*(num_rows-1.0)-top_offset-bottom_offset)
#	size_h = 1./float(fig_per_line)*(1.0-margin*(fig_per_line-1.0)-left_offset-right_offset)
#	#print 'size = ',size

#	# initialize a list of subplots to append to
#	subplots = []

#	# dynamically create list of coordinates of each subimage
#	for line in range(num_rows):
#		for col in range(fig_per_line):
#			subplots.append([left_offset+margin*(col+1.0)+size_h*col,top_offset+margin*(num_rows-line)+size_v*(num_rows-line-1.0),size_h,size_v])
#		
	gs = gridspec.GridSpec(num_rows,fig_per_line)
	gs.update(wspace=0.0,hspace=0.0)
	if margin != None:
		gs.update(left=left_offset,right=1.0-right_offset,top=1.0-top_offset,bottom=bottom_offset,wspace=margin,hspace=margin)

	sources = sourcetable.group_by('Cluster')
	for key,clustertable in zip(sources.groups.keys,sources.groups):
		if clustertable['Cluster'][0] == cluster:
			for n in range(N):
				showy=top=right=False
				showx=True
				if n % fig_per_line ==0:
					showy = True
				if n % fig_per_line == fig_per_line-1:
					right=True
				if n < fig_per_line:	
					top=True
				show_axes=[showx,showy,top,right]
				source = Table(clustertable[n])
				ax = plt.subplot(gs[n])
				ax.set_xticks([])
				ax.set_yticks([])

				#ax = figure.add_axes(subplots[n])
				markerPlotSEDax(ax,source,show=False,folder_export=folder_export,RAstr="RA",DECstr="DEC",show_axes=show_axes)
	#for key,sourcetable in zip(sources.groups.keys,sources.groups):
	#	if sourcetable['Cluster'][0]==cluster and sourcetable['Property'][0] != 'Extended' and sourcetable['Property'][0] != 'Total_Cluster':
	figure.savefig(folder_export+cluster+"_SEDs.png",dpi=300)
	if show: plt.show()


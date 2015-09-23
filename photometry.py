### This scrits attempts to measure the fluxes of the point sources in the
### SOFIA fields
### MJ Rizzo 2014-2015


from astropy import wcs
from astropy.io import fits
from astropy.table import Table,vstack,hstack,Column
from astropy import units as u
from astropy import constants

from numpy import ravel
import numpy as np
import time
from itertools import izip

from photutils.morphology import centroid_com as centroid
from photutils.morphology import fit_2dgaussian
import photutils as phot
from photutils.aperture_core import _sanitize_pixel_positions
from photutils import utils
from imgproc import calc_bkg_rms,calc_masked_aperture
import matplotlib.pyplot as plt
import pickle
import re
import os
from astropy.coordinates import SkyCoord,FK5
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar,newton

from matplotlib import rcParams
import seaborn as sns
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
#rcParams['text.usetex'] = True
sns.set_style('whitegrid')


def do_aperture_phot_cal(hdu,boxsize=5,r_ap=4):
	'''
	Does aperture photometry on a calibrator
	'''
	# load image
	img = hdu[0].data

	# deals with image when it is given as a cube
	if len(img.shape)==3:
		img = img[0,:,:]
	mask = np.isnan(img)
	# reads header
	prihdr = hdu[0].header
	PROCSTAT = prihdr['PROCSTAT']
	flight = prihdr['MISSN-ID']
	wavelength = prihdr['WAVELNTH']
	dichroic = prihdr['DICHR_S']
	el1 = prihdr['SPECTEL1']
	el2 = prihdr['SPECTEL2']
	DATE_TIME = prihdr['DATE-OBS']
	cal_name = prihdr['OBJECT']
	if "Barr" in dichroic:
		dichroic = "Dichroic"
	if "Open" in dichroic:
		dichroic = "Open"
	X = prihdr['ST0X']
	Y = prihdr['ST0Y']
	r_in = prihdr['STARBCK1']
	r_out = prihdr['STARBCK2']
	starap = prihdr['STARAP']
	ST0F = prihdr['ST0F']
	date,time = DATE_TIME.split('T')

	# Initialize lists
	flux = []
	areas = []
	EEflux = []
	EEareas = []
	
	# Annulus photometry for background subtraction
	EEannulus = phot.CircularAnnulus((X,Y),r_in,r_out)

	# Calculate the encircled energy at different radii
	radii = np.arange(0.5,15,0.5)
	for rad in radii:
		aperture = phot.CircularAperture((X,Y),r=rad)
		EEareas.append(aperture.area())
		EEflux.append(phot.aperture_photometry(img,aperture,mask=mask))
	radial_profile = vstack(EEflux)
	radial_areas = np.array(EEareas)

	# calculate the local background in the large annulus using mmm
	bkgd_surf_brightness, area = calc_masked_aperture(EEannulus, img, method='mmm', mask=mask)
	
	# calculate the noise in the annulus
	bkg_rms,bkg_aps = calc_bkg_rms(EEannulus, img, src_ap_area = np.pi*r_ap**2,rpsrc = r_ap, mask=None, min_ap=6)

	# subtract background
	background = bkgd_surf_brightness*radial_areas
	radial_array = np.array(radial_profile['aperture_sum'])
	radial_array = radial_array -  background

	# interpolate the encircled energy distribution
	EEfunc = interp1d(radii,radial_array/np.amax(radial_array),kind='quadratic')

	# try to solve for R50%, the full width half max of encircled energy
	try:
		FWHS = newton(lambda x:EEfunc(x)-0.5,4)
	except:
		FWHS = 0.0
	
	# r_ap is the normal aperture size in pixels that we will use at all wavelengths
	for rad in (r_ap,starap):
		aperture = phot.CircularAperture((X,Y),r=rad)
		areas.append(aperture.area())
		flux.append(phot.aperture_photometry(img,aperture,mask=mask))
	aper_table = hstack(flux)

	
	# compares with the already-existing fit (to a Moffat profile) in the FITS file header, if available
	moffat_corr=0.0
	try:
		ST0PF = prihdr['ST0PF']
		moffat_flux = ST0PF
		moffat_corr = (moffat_flux)/(aper_table['aperture_sum_1'][0]-bkgd_surf_brightness*areas[0])
	except:
		#print "No Moffat profile fitted for this calibrator"
		pass		
	
	# estimate the flux from the calibrator over the large aperture starap
	photometry = aper_table['aperture_sum_2'][0] - bkgd_surf_brightness*areas[1]
	
	# calculate the difference from the existing fits
	percent = ((ST0F - photometry)/ST0F)*100
	
	# Estimate the SNR
	SNR = photometry/bkg_rms
	
	# define the aperture correction to be larger than 1 
	aper_corr = (photometry)/(aper_table['aperture_sum_1'][0]-bkgd_surf_brightness*areas[0])
	
	# record noise level
	sensitivity = bkg_rms
	
	return flight,wavelength,aper_corr,cal_name,starap,percent,SNR,moffat_corr,date,time,dichroic,photometry,el1,el2,PROCSTAT,FWHS,sensitivity


def make_calibrator_table(calfolder="/cardini3/mrizzo/2012SOFIA/2014/Calibrators/",folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"):
	'''
	this function creates a table with the aperture correction for calibrators of each flight
	'''
	# prep table & formats
	caltable = Table(names=("Flight_ID","Date","Time","Filename", "Cal_name", "Dichroic", 				"Wavelength","Ap_Size","Aper_corr","Diff_pipe","SNR","Moffat_corr","Phot_val","El1","El2","ProcStat","R50","sensitivity"),\
			dtype=('<S30','<S30','<S30','<S100','<S30','<S30','<f8','<i8','<f8','<f8','<f8','<f8','<f8','<S30','<S30','<S30','<f8','<f8'))
	caltable['Aper_corr'].format = "1.3f"
	caltable["Diff_pipe"].format = "1.3f"
	caltable["SNR"].format = "3.3f"
	caltable["Moffat_corr"].format = "1.3f"
	caltable["Phot_val"].format = "3.3f"
	caltable["R50"].format = "3.3f"
	caltable["sensitivity"].format = "3.3f"

	# explore folder with all calibrators
	for root,dir,filenames in os.walk(calfolder):
		for filename in filenames:
			if "UND" not in filename:
				# open FITS file
				filepath = os.path.join(root,filename)
				hdu = fits.open(filepath)
				
				# calculate aperture correction & other calibrator properties
				flight,wavelength,aper_corr,cal_name,starap,percent,SNR,moffat,date,\
						time,dichroic,phot_val,el1,el2,procstat,fwhs,sensitivity = do_aperture_phot_cal(hdu)

				# add to table
				caltable.add_row([flight,date,time,"/".join(filepath.split("/")[7:]),cal_name,dichroic,wavelength,\
						starap,aper_corr,percent,SNR,moffat,phot_val,el1,el2,procstat,fwhs,sensitivity])
	
	# print & export
	print len(caltable)," calibrator observations found"
	pickle.dump(caltable,open(folder_export+"caltable.data","wb"))
	caltable.write(folder_export+"caltable.txt",format="ascii.fixed_width")

def make_calibrator_corr_table(folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"):
	''' 
	This function calculates the ratio between the "Open" and the "Dichroic" modes in the 37um channel, for all flights.
	It uses the average of all flights in case the data is missing.
	'''

	# Load and init
	caltable = pickle.load(open(folder_export+"caltable.data","r"))
	wl = 37.1
	openlbl = "Open"
	opencaltable = Table(names=['Flight_ID','Cal_corr','Cal_name','Open_rms','Dich_rms'],dtype=['S50','f8','S50','f8','f8'])
		
	# divide calibrator table into different flights
	flights = caltable.group_by('Flight_ID')
	for key,group in izip(flights.groups.keys,flights.groups):
		
		# group by wavelength
		wavelengths = group.group_by('Wavelength')
		for wlkey,wlgroup in izip(wavelengths.groups.keys,wavelengths.groups):
			
			# check if this is the 37 micron band
			if wl == wlkey['Wavelength']:
				# group by calibrator name
				calibgroup = wlgroup.group_by('Cal_name')
				for calkey,calgroup in izip(calibgroup.groups.keys,calibgroup.groups):
					
					# init variables
					opencal=dichcal=opencalrms=dichcalrms=0.0

					# group by dichroic setting
					dichroic_group = calgroup.group_by('Dichroic')
					for dichkey,dichgroup in izip(dichroic_group.groups.keys,dichroic_group.groups):
						#print dichgroup['Flight_ID','Cal_name','Phot_val','Dichroic','El1','El2']
						if dichkey['Dichroic']=='Open':
							opencal = np.mean(dichgroup['Phot_val'])
							opencalrms = np.std(dichgroup['Phot_val'])
						elif dichkey['Dichroic']=='Dichroic':
							dichcal = np.mean(dichgroup['Phot_val'])
							dichcalrms = np.std(dichgroup['Phot_val'])
					
					# calculate ratio of open/dichroic for that flight and that calibrator
					if opencal!=0.0 and dichcal!=0.0:
						opencaltable.add_row([key['Flight_ID'],opencal/dichcal,calkey['Cal_name'],opencalrms,dichcalrms])

	# form a list of all calibrators in all flights
	cal_corr = [opencaltable['Cal_corr'][i] for i in range(len(opencaltable)) if opencaltable['Cal_corr'][i]>0.0 \
			if opencaltable['Cal_corr'][i]!=np.inf]
	
	# select only calibrators
	finaltable = opencaltable['Flight_ID','Cal_corr']

	# fill in for the two flights that have no Open calibrator; use the mean of all other calibrators for all flights
	finaltable.add_row(['2013-09-12_FO_F129',np.mean(cal_corr)])
	finaltable.add_row(['2013-06-26_FO_F109',np.mean(cal_corr)])

	# save & export
	pickle.dump(finaltable,open(folder_export+"cal_corr.data",'wb'))
	finaltable.write(folder_export+"cal_corr.txt",format='ascii.fixed_width')	



def plot_cal_profiles(calfolder="/cardini3/mrizzo/2012SOFIA/2014/Calibrators/",\
	folder_import = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/", \
	folder_export = "/cardini3/mrizzo/2012SOFIA/2014meta/Calibrators/"):
	''' this function creates azimuthally averaged radial profiles of calibrator stars'''
	
	# loads up
	caltable = pickle.load(open(folder_import+"caltable.data",'r'))

	# set the range of radii to calculate the profile on
	radii = np.arange(0.5,15,0.5)

	# Plot all profiles for a given wavelength & dichroic setting, with error bars corresponding to the variation across flights and sources
	wavelength_group = caltable.group_by('Wavelength')

	# create one image for the average of all EE profiles for each wavelength
	figw,axw = plt.subplots()
	for wlkey,wlgroup in izip(wavelength_group.groups.keys,wavelength_group.groups):
		wavelength = wlkey['Wavelength']
		dich_mode = wlgroup.group_by('Dichroic')
		fig,ax = plt.subplots()	
		for dichkey,dichgroup in izip(dich_mode.groups.keys,dich_mode.groups):
			dichroic = dichkey['Dichroic']
			L = len(dichgroup)

			# create one image per setting
			fig,ax = plt.subplots()
			if (wavelength in [11.1,19.7,31.5] and dichroic=="Dichroic") or wavelength == 37.1:
				for i in range(L):
					calibrator = dichgroup[i]
					flight = calibrator['Flight_ID']
					# don't bother with this bad flight
					if flight != '2014-05-02_FO_F166': 
						# loads up the fits file & reads the header
						calname = calibrator['Cal_name']
						filename = os.path.join(calfolder,"sofia_"+flight,calibrator["Filename"])
						hdu = fits.open(filename)
						img = hdu[0].data
						#mask = np.isnan(img)
						prihdr = hdu[0].header
						if len(img.shape)==3:
							img = img[0,:,:]
						X = prihdr['ST0X']
						Y = prihdr['ST0Y']
						r_in = prihdr['STARBCK1']
						r_out = prihdr['STARBCK2']
						flux = []
						areas = []

						# Set up background aperture & calculate background using mmm
						annulus = phot.CircularAnnulus((X,Y),r_in,r_out)
						annulus_area = annulus.area()
						bkgd, area = calc_masked_aperture(annulus, img, method='mmm', mask=None)

						# create radial profile using this background measurement
						for rad in radii:
							aperture = phot.CircularAperture((X,Y),r=rad)
							areas.append(aperture.area())
							flux.append(phot.aperture_photometry(img,aperture))
						radial_profile = vstack(flux)
						radial_areas = np.array(areas)
						background = bkgd*radial_areas
						radial_array = np.array(radial_profile['aperture_sum'])
						radial_array = radial_array -  background

						# normalize the encircled energy to 1
						radial_array /= np.amax(radial_array)

						# plot on current plot
						ax.plot(radii,radial_array,color='gray',alpha=0.5)
						if i==0:
							radial = radial_array.copy()
						else:				
							radial = np.column_stack((radial,radial_array))
				
				# doing some cleanup
				np.delete(radial,0,0)

				# set up the plot
				ax.set_title(str(wavelength)+"_"+dichroic)
				ax.grid(True)
				ax.set_xlabel("Aperture radius (pixels)")
				ax.set_ylabel("Normalized encircled energy (Jy)")
				fig.savefig(folder_import+str(wavelength)+"_"+dichroic+".png")
				plt.close(fig)
				
				# print with error bar, using the std of all values for errors
				# this is plotted on the average plot
				axw.errorbar(radii,np.mean(radial,axis=1),yerr=[np.std(radial,axis=1),\
					np.std(radial,axis=1)],linewidth=2,label=str(wavelength)+"_"+dichroic,alpha=0.7)
	
	# set up average plot	
	axw.set_title("average")
	axw.legend(loc=4)
	axw.grid(True)
	axw.set_ylim([0,1.02])
	axw.set_xlabel("Aperture radius (pixels)")
	axw.set_ylabel("Normalized encircled energy (Jy)")
	figw.savefig(folder_import+"average.png")
	#plt.show()
	#print flight,wavelength,np.std(radial,axis=1),radial


def parse_new_source_file(filename,folder_export="/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"):
	'''
	This function parses a DS9 regions file and populates a source table 
	'''
	# open the file and create an empty file with header
	f = open(filename,'r')
	f_out = open(folder_export+"all_sources_names.reg",'w')
	f_out.write("# Region file format: DS9 version 4.1\n")
	f_out.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
	f_out.write('fk5\n')

	# this defines & initializes the new master table
	table = Table(names=("Cluster","SOFIA_name", "RA", "DEC", "Property","Type","SemiMajorAxis","SemiMinorAxis","Angle"),\
			dtype=('<S15','<S15','<f8','<f8','<S15','<S15','<f8','<f8','<f8'),masked=True)
	table_bkgd = Table(names=("Cluster","SOFIA_name", "RA", "DEC", "Property","Type","SemiMajorAxis","SemiMinorAxis","Angle"),\
			dtype=('<S15','<S15','<f8','<f8','<S15','<S15','<f8','<f8','<f8'),masked=True)
	sourcecount=0
	prevcluster = ""
	cluster_number = 0

	# read file by skipping the DS9 header
	for line in f.readlines()[3:]:
		
		splitstr = line.split()
		cluster = splitstr[-1]
		DEC = splitstr[0].split(",")[1]
		RA = splitstr[0].split(",")[0].split("(")[1]
		Type = splitstr[0].split(",")[0].split("(")[0]

		# Sets properties depending on colors in DS9 regions file
		if "white" in line:
			prop = "Extended"
		elif "red" in line:
			prop = "Clustered"
		elif "cyan" in line:
			prop = "Total_Cluster"
		else:
			prop = "Isolated"

		# Read in Ellipse parameters
		if Type=="ellipse":
			semimaj = splitstr[0].split(",")[2].split('"')[0]
			semimin = splitstr[0].split(",")[3].split('"')[0]
			angle = splitstr[0].split(",")[4].split(')')[0]
		else:
			semimaj = splitstr[0].split(",")[2].split('"')[0]
			semimin = semimaj
			angle = 0

		# Deals with the background field, populate background table
		if "background" in line:
			cluster_number=0
			prevcluster =cluster
			SOFIA_name = cluster+"_bkgd"
			Type = "bkgd"
			table_bkgd.add_row([cluster,SOFIA_name,RA,DEC,prop,Type,semimaj,semimin,angle])
		else:
			# Populate source table
			cluster_number += 1
			SOFIA_name = cluster+"."+str(cluster_number)
			table.add_row([cluster,SOFIA_name,RA,DEC,prop,Type,semimaj,semimin,angle])

		# write a proper DS9 region file to be read later
		f_out.write(" ".join(splitstr[:-2]))

		# add in the cluster names on top of each aperture
		if "#" in splitstr[:-2]:
			#print splitstr[:-2]
			f_out.write(' text={'+SOFIA_name+'}\n')
		else:
			f_out.write(' # text={'+SOFIA_name+'}\n')
	print table
	print table_bkgd
	f.close()
	f_out.close()
	return table,table_bkgd

def get_from_header(header,keyword):
	'''
	Extracts Header info from SOFIA files that have been modified my MIRIAD
	Header keyword has to be right after 'HISTORY'
	'''
	headerlist =  header['HISTORY']
	for line in headerlist:
		if keyword in line:
			return re.sub(r"'",'',(re.sub(r'\s','',line).split('=')[1].split("/")[0]))


def get_aper_corr(hdu,wavelength,folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"):
	''' 
	finds the corresponding aperture correction given a file and a wavelength
	 '''
	# load up the calibrator tables
	caltable = pickle.load(open(folder_export+"caltable.data",'r'))

	# load up the header from the provided image
	header = hdu[0].header

	# extract flight, dichroic and element infos from header
	flight = get_from_header(header,"MISSN-ID")
	dichroic = get_from_header(header,"DICHR_S")
	el1 = get_from_header(header,"SPECTEL1")
	el2 = get_from_header(header,"SPECTEL2")
	
	# parse dichroic
	if "Barr" in dichroic:
		dichroic = "Dichroic"
	if "Open" in dichroic:
		dichroic = "Open"

	# prepare to iterate on all the flights
	flights = caltable.group_by('Flight_ID')
	val=0
	rms=0

	# all wavelengths
	if wavelength == 11:
		wl=11.1
	elif wavelength == 19:
		wl = 19.7
	elif wavelength == 31:
		wl = 31.5
	else:
		wl = 37.1
	for key,group in izip(flights.groups.keys,flights.groups):
		if flight in key['Flight_ID']:
			mode = group.group_by('Dichroic')
			for modekey,modegroup in izip(mode.groups.keys,mode.groups):
				if dichroic in modekey['Dichroic']:
					obs = modegroup.group_by('Wavelength')
					for wlkey,wlgroup in izip(obs.groups.keys,obs.groups):
						if wl == wlkey['Wavelength']:
							# calculate the mean and std of the aperture corrections for this flight, dichroic, and wavelength
							val = np.mean(wlgroup['Aper_corr'])
							rms = np.std(wlgroup['Aper_corr'])

	# if the value is 0, it means that the data is missing - we use instead the mean of all other flights as a proxy for the aperture corrections
	if val==0:
		obs = caltable.group_by('Wavelength')
		for wlkey,wlgroup in izip(obs.groups.keys,obs.groups):
			if wl == wlkey['Wavelength']:
				mode = wlgroup.group_by('Dichroic')
				for modekey,modegroup in izip(mode.groups.keys,mode.groups):
					if dichroic in modekey['Dichroic']:			
						val = np.mean(modegroup['Aper_corr'])
						rms = np.std(modegroup['Aper_corr'])
	return val,rms

def photBkgd(sourcetable, \
		fitsfolder="/cardini3/mrizzo/2012SOFIA/alldata/",\
		folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/",\
		metafolder = "/cardini3/mrizzo/2012SOFIA/2014meta/"):
	'''
	This function determines the background statistics, using the field defines as the background in each frame.
	'''
	# start fresh
	allrows = Table()

	# wavelengths
	wllist = [11,19,31,37]

	# nominal aperture size used for photometry
	r_out=4

	# loop on the sources in the table
	for i in range(len(sourcetable)):
		line = sourcetable[i]
		newRow=line
		t=Table()
		
		# read in name and position
		cluster = line['Cluster']
		RA = line['RA']
		DEC = line['DEC']

		# loop on all wavelengths
		for wl in wllist:
			# construct mosaic name
			mosaic_name = cluster+"."+str(wl)

			# browse through file directory - could choose to use the crop.fits files, or the crop.subtracted.fits files
			dirlist = os.listdir(fitsfolder)

			# form the list of available files which are relevant to us
			filelist = [f for f in dirlist if "crop.fits" in f if mosaic_name in f and "mask" not in f]

			# loads the exposure times and sensitivities estimated during a previous step of reduction
			meta_list=Table.read(metafolder+mosaic_name+'.txt',format='ascii')

			# when we start, we have zero fields found
			nFields=0				

			# loop on all the files
			for f in filelist:
				# loading up image, header, and WCS info in header
				hdu = fits.open(fitsfolder+f)
				prihdr = hdu[0].header
				w = wcs.WCS(hdu[0].header)

				# image data
				img = hdu[0].data

				# Masking
				image = np.ma.masked_invalid(img)
				mask = image.mask

				# Converts RA and DEC to pixel coordinates
				X,Y = w.wcs_world2pix(RA,DEC,1)

				# this is the radius of the aperture used for background determination
				rad = line['SemiMajorAxis']/0.768

				# if source is within image bounds and not masked out, then proceed to photometry
				if (X>rad and X<image.shape[1]-rad) and (Y>rad and Y<image.shape[0]-rad) \
						and not image.mask[Y-rad:Y+rad,X-rad:X+rad].any():
					
					# determine the aperture photometry from previous script
					aper_corr,tmp = get_aper_corr(hdu,wl)

					# we have found another field in which the source is!
					nFields+=1
					field = f.split(".")[2]

					# print out things to make user happy
					print "Source "+line['SOFIA_name']+" was found at "+str(wl)+"um in field "+field+" of "+mosaic_name

					# define the background as an annulus of 1/3 its radius out to its radius
					ap = phot.CircularAnnulus((X,Y),r_in=rad/3.,r_out=rad)

					# flux in that annulus
					flux = phot.aperture_photometry(image,ap,mask = mask)

					# area of a circular aperture used for photometry
					src_ap_area = np.pi*r_out**2

					# calculate the background rms
					bkg_rms,bkg_aps = calc_bkg_rms(ap, image, src_ap_area, rpsrc = r_out, mask=mask, min_ap=6)

					# update table with flux
					t["Flux_"+str(wl)+"_"+str(nFields)] = flux["aperture_sum"]*r_out**2/ap.area()*aper_corr
					t["Flux_"+str(wl)+"_"+str(nFields)].format = "4.3f"	

					# update table with flux error
					t["RMS_"+str(wl)+"_"+str(nFields)]=bkg_rms*aper_corr
					t["RMS_"+str(wl)+"_"+str(nFields)].format= "4.3f"

			# check if we found multiple fields (only the case for the 37 micron band)
			if nFields==2:
				t["RMS_tot_"+str(wl)] = 1./np.sqrt(1./t["RMS_"+str(wl)+"_1"][0]**2+1./t["RMS_"+str(wl)+"_2"][0]**2)
			else:
				t["RMS_tot_"+str(wl)] = t["RMS_"+str(wl)+"_1"][0]

		# add those results to the existing source table
		if nFields!=0:
			newRow = hstack([newRow,t])

		# now that this is done for all wavelenths, we need to add it to a table
		if i==0:
			allrows=newRow
		else:
			allrows = vstack([allrows,newRow])
	return allrows


def calcR50(image,mask,bkg_ap,radii,bkgd_subtract=True):
	'''
	Calculates the half width half max of encircled energy distribution
	'''
	# load image data
	img=image.data
	
	# empty lists to start
	EEflux = []
	EEareas = []

	# Annulus photometry for the background subtraction
	EEbkgd, bkg_area = calc_masked_aperture(bkg_ap, image, method='mmm',mask=mask)

	# do photometry for a bunch of radii
	for rad in radii:
		aperture = phot.CircularAperture(bkg_ap.positions,r=rad)
		EEareas.append(aperture.area())
		EEflux.append(phot.aperture_photometry(img,aperture,mask=mask))

	# stack the results
	radial_profile = vstack(EEflux)
	radial_areas = np.array(EEareas)

	# evaluate background equivalent flux in each aperture
	background = EEbkgd*radial_areas

	# this is the radial profile, first with just the fluxes
	radial_array = np.array(radial_profile['aperture_sum'])

	# then with the fluxes minus the background; user can choose not to subtract background.
	if bkgd_subtract:
		radial_array = radial_array -  background
	
	# create a Encircled Energy function by interpolating between the radii
	EEfunc = interp1d(radii,radial_array/np.amax(radial_array),kind='quadratic')
	
	# try to find the half way point
	try:
		R50 = newton(lambda x:EEfunc(x)-0.5,4)
	except:
		R50 = 0.0

	return R50

def get_cal_quantity(hdu,wavelength,quantity,folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"):
	'''
	finds the corresponding calibrator quantity given a file and a wavelength
	function is inspired by 'get_aper_corr' function
	'''
	# Load calibrator file
	caltable = pickle.load(open(folder_export+"caltable.data",'r'))
	header = hdu[0].header

	# load parameters
	flight = get_from_header(header,"MISSN-ID")
	dichroic = get_from_header(header,"DICHR_S")
	el1 = get_from_header(header,"SPECTEL1")
	el2 = get_from_header(header,"SPECTEL2")
	if "Barr" in dichroic:
		dichroic = "Dichroic"
	if "Open" in dichroic:
		dichroic = "Open"
	print flight, dichroic
	flights = caltable.group_by('Flight_ID')
	val=0
	rms=0
	if wavelength == 11:
		wl=11.1
	elif wavelength == 19:
		wl = 19.7
	elif wavelength == 31:
		wl = 31.5
	else:
		wl = 37.1

	# search for given quantity
	for key,group in izip(flights.groups.keys,flights.groups):
		if flight in key['Flight_ID']:
			mode = group.group_by('Dichroic')
			for modekey,modegroup in izip(mode.groups.keys,mode.groups):
				if dichroic in modekey['Dichroic']:
					obs = modegroup.group_by('Wavelength')
					for wlkey,wlgroup in izip(obs.groups.keys,obs.groups):
						if wl == wlkey['Wavelength']:
							val = np.mean(wlgroup[quantity])
							rms = np.std(wlgroup[quantity])
	if val==0:
		obs = caltable.group_by('Wavelength')
		for wlkey,wlgroup in izip(obs.groups.keys,obs.groups):
			if wl == wlkey['Wavelength']:
				mode = wlgroup.group_by('Dichroic')
				for modekey,modegroup in izip(mode.groups.keys,mode.groups):
					if dichroic in modekey['Dichroic']:			
						val = np.mean(modegroup[quantity])
						rms = np.std(modegroup[quantity])
	return val,rms


def newPhot(sourcetable, \
		fitsfolder="/cardini3/mrizzo/2012SOFIA/alldata/",\
		folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/",\
		metafolder = "/cardini3/mrizzo/2012SOFIA/2014meta/",\
		phot_corr=[]):
	'''
	This is the main photometry function
	'''
	# start fresh
	allrows = Table(masked=True)

	# wavelengths
	wllist = [11,19,31,37]

	# nominal aperture size used for photometry
	r_out=4

	# load the background table used before, to know the sensitivity limits
	bkgd_table = pickle.load(open(folder_export+"photbkgd.data","r"))

	# loop on the sources in the table
	for i in range(len(sourcetable)):
		
		# read in the existing data, extraact cluster name
		line = sourcetable[i]
		newRow=line
		cluster = line['Cluster']

		# start with an empty table for the photometry results
		t=Table(masked=True)
		
		# form a table with the sensitivity limits
		for j in range(len(bkgd_table)):
			if bkgd_table["Cluster"][j] == cluster:
				bkgdline = Table(bkgd_table[j])

		# read source location
		RA = line['RA']
		DEC = line['DEC']

		# loop for each wavelength
		for wl in wllist:

			# construct mosaic name
			mosaic_name = cluster+"."+str(wl)

			# browse through file directory - could choose to use the crop.fits files, or the crop.subtracted.fits files
			dirlist = os.listdir(fitsfolder)

			# form the list of available files which are relevant to us
			filelist = [f for f in dirlist if "crop.fits" in f if mosaic_name in f and "mask" not in f]

			# loads the exposure times and sensitivities estimated during a previous step of reduction
			meta_list=Table.read(metafolder+mosaic_name+'.txt',format='ascii')

			# we haven't found any field yet
			nFields=0		
		
			# loop on all the files
			for f in filelist:

				# loading up image, header, and WCS info in header
				hdu = fits.open(fitsfolder+f)
				prihdr = hdu[0].header
				w = wcs.WCS(hdu[0].header)

				# retrieve flight info
				flight = get_from_header(prihdr,"MISSN-ID")

				# image data
				img = hdu[0].data

				# Masking
				image = np.ma.masked_invalid(img)
				mask = image.mask

				# Converts RA and DEC to pixel coordinates
				X,Y = w.wcs_world2pix(RA,DEC,1)

				# find aperture correction
				aper_corr,tmp = get_aper_corr(hdu,wl)

				# set apertures depending on the type of photometry
				if line["Property"] == "Isolated":
					# aperture used for photometry
					ap = phot.CircularAperture((X,Y),r=r_out)

					# aperture used for background subtraction and sensitivity calculation
					bkg_ap = phot.CircularAnnulus((X,Y),3*r_out,5*r_out)

					# value used for aperture correction
					apcor = aper_corr

				elif line["Property"] == "Extended" or line["Property"] == "Total_Cluster":
					
					# read ellipse properties
					a = line['SemiMajorAxis']/0.768
					b = line['SemiMinorAxis']/0.768
					angle = line['Angle']*np.pi/180.

					# create apertures
					ap = phot.EllipticalAperture((X,Y),a,b,angle)
					bkg_ap = phot.EllipticalAnnulus((X,Y),a_in=a,a_out=a*1.2,b_out=b*1.2,theta=angle)

					# no aperture correction when measuring extended sources
					apcor = 1.

				elif line["Property"] == "Clustered":
					ap = phot.CircularAperture((X,Y),r=r_out)

					# very large annulus when looking at clustered sources
					bkg_ap = phot.CircularAnnulus((X,Y),6*r_out,10*r_out)
					apcor = aper_corr

				#if source is within image bounds and not masked out
				if (X>r_out and X<image.shape[1]-r_out) and (Y>r_out and Y<image.shape[0]-r_out) \
						and not image.mask[Y-r_out:Y+r_out,X-r_out:X+r_out].any():
					
					print "Aperture correction: ",aper_corr

					# we found a field!
					nFields+=1

					# Field number is third entry in filename
					field = f.split(".")[2]
				
					# print out background sensitivity for sanity check
					bkgd_sensitivity = bkgdline["RMS_tot_"+str(wl)][0]
					print 'Background Sensitivity:',bkgd_sensitivity	
					(est_sensitivity,) = [meta_list[k] for k in range(len(meta_list))\
								if meta_list[k]["Field name"] == mosaic_name+"."+field]	

					# Calculate the flux in image			
					flux = phot.aperture_photometry(image*~mask,ap,mask = mask)

					# Calculate the background
					bkg, bkg_area = calc_masked_aperture(bkg_ap, image*~mask, method='mmm', mask=mask)
					
					# Calculate the sensitivity
					bkg_rms, bkg_aps = calc_bkg_rms(bkg_ap, image*~mask,src_ap_area = np.pi*r_out**2, rpsrc = r_out, mask=mask, min_ap=6)

					# update flux field in table
					t["Flux_"+str(wl)+"_"+str(nFields)] = (flux["aperture_sum"][0] - bkg*ap.area())*apcor
					t["Flux_"+str(wl)+"_"+str(nFields)].format = "4.3f"

					# print to make user happy
					print 'flux: ',flux["aperture_sum"][0], 'bkg:', bkg,'area: ',ap.area()

					# three-way sensitivity calculations for inspection; only first one is used
					t["Sensitivity_calc_"+str(wl)+"_"+str(nFields)]= bkg_rms*apcor#bkgd_sensitivity
					t["Sensitivity_calc_"+str(wl)+"_"+str(nFields)].format = "4.3f"

					# theoretical sensitivity from SOFIA online calculator and exposure times
					t["Sensitivity_theo_"+str(wl)+"_"+str(nFields)]=est_sensitivity['Sensitivity']/5.
					t["Sensitivity_theo_"+str(wl)+"_"+str(nFields)].format = "4.3f"

					# sensitivity calculated from full background estimation at the early reduction stage (MIRIAD)
					t["Sensitivity_calc2_"+str(wl)+"_"+str(nFields)]=est_sensitivity['Sensitivity_meas']*aper_corr
					t["Sensitivity_calc2_"+str(wl)+"_"+str(nFields)].format = "4.3f"					

					# print sensitivities for inspection
					print "1-sigma Sensitivity calc/theo/calc2: ",t["Sensitivity_calc_"+str(wl)+"_"+str(nFields)][0],\
						t["Sensitivity_theo_"+str(wl)+"_"+str(nFields)][0],t["Sensitivity_calc2_"+str(wl)+"_"+str(nFields)][0]


					# Calculate half width half max of encircled energy distribution for source and calibrator
					radii=np.arange(0.5,15,0.5)	
					t["R50_"+str(wl)] = calcR50(image*~mask,mask,bkg_ap,radii,bkgd_subtract=True)
					t["R50_"+str(wl)].format = "2.3f"
					R50_cal,rms = get_cal_quantity(hdu,wl,'R50')
					t["R50_cal_"+str(wl)] = R50_cal
					t["R50_cal_"+str(wl)].format = "2.3f"

					# add filename, flight to table
					t["fields_"+str(wl)+"_"+str(nFields)] = f
					t["Flight_ID_"+str(wl)+"_"+str(nFields)] = flight
					t["fields_"+str(wl)+"_"+str(nFields)].unit = u.dimensionless_unscaled
					t["Flight_ID_"+str(wl)+"_"+str(nFields)].unit = u.dimensionless_unscaled
		
					# add exposure time, calibration factor
					t["ExpTime_"+str(wl)+"_"+str(nFields)]=est_sensitivity["Exposure time"]
					t["ExpTime_"+str(wl)+"_"+str(nFields)].unit=u.second
					t["Calfctr_"+str(wl)+"_"+str(nFields)]=est_sensitivity["Cal Factor"]
					t["Calfctr_"+str(wl)+"_"+str(nFields)].unit=u.dimensionless_unscaled
					t["Calfctr_"+str(wl)+"_"+str(nFields)].format="4.5f"

			# consolidate all flux estimates, weighing by the sensitivities
			aveflux = 0.0;tottime = 0.0;inttime = 0.0;calftcr = 1.0;tot_weight = 0.0;tot_time = 0.0
			flightstr = ""
			if nFields==0:
				print "Source "+line['SOFIA_name']+" was not found at "+str(wl)+"um in any field "
			else:
				# for all fields found
				for j in range(nFields):
					if j==0:
						# initialize the calibration factor and the integration time for the first field
						calfctr = t["Calfctr_"+str(wl)+"_"+str(j+1)][0]
						inttime = t["ExpTime_"+str(wl)+"_"+str(j+1)][0]

					# it means that this source was found in more than one SOFIA image
					else:
						# the effective integration time for that source is a combination of the exposure times,
						# adjusted for the fact that the calibration factor changes.
						inttime = (t["Calfctr_"+str(wl)+"_"+str(j+1)][0]/calfctr) * t["ExpTime_"+str(wl)+"_"+str(j+1)][0]

					# add the flight name if not already existing
					if t["Flight_ID_"+str(wl)+"_"+str(j+1)][0] not in flightstr:					
						flightstr += t["Flight_ID_"+str(wl)+"_"+str(j+1)][0]+" "

					# calculate the weight of this observation, based on the sensitivity
					invsens = 1./(t["Sensitivity_calc_"+str(wl)+"_"+str(j+1)][0]**2)

					# add the new flux with the appropriate weight, and stack the calfactor-weighted integration time
					if not np.isnan(t["Flux_"+str(wl)+"_"+str(j+1)][0]):
						aveflux +=  invsens * t["Flux_"+str(wl)+"_"+str(j+1)][0]
						tot_weight += invsens
						tottime += inttime

					# if there are issues, then mask this
					else:
						t.mask["Flux_"+str(wl)+"_"+str(j+1)][0] = True

				# when we have unmasked results
				if tot_weight != 0.0:

					# the photometry is the weighted sum of fluxes, divided by the weights
					t["F"+str(wl)] = aveflux/tot_weight

					# set nominal flag
					t["flag_F"+str(wl)] = 'N'

					# when flux is smaller than the background RMS (sensitivity floor), then flag the data and set the flux to the sensitivity floor
					if t["F"+str(wl)][0]< bkgdline["RMS_tot_"+str(wl)][0]:
						t["F"+str(wl)][0] = bkgdline["RMS_tot_"+str(wl)][0]
						# 'U' for upper limit
						t["flag_F"+str(wl)][0] = 'U'
					t["F"+str(wl)].format = "4.3f"
				

					# this is the adopted error on each flux estimate, mostly due to observatory variations; 15% error plus whatever local error there is
					t["e_F"+str(wl)] = np.sqrt(1./(tot_weight+1./(0.15*t["F"+str(wl)][0])**2))
					t["e_F"+str(wl)].format = "4.3f"

					# few more checks to make sure we get valid data
					if t["F"+str(wl)][0]<0 or np.isnan(t["F"+str(wl)][0]):
						t.mask["F"+str(wl)][0] = True
						t.mask["e_F"+str(wl)][0] = True
						t.mask["flag_F"+str(wl)][0] = True

					# this is the 1-sigma combined sensivity for this source, without the 15% systematic error
					t["Sensitivity_calc_"+str(wl)] = np.sqrt(1./tot_weight)

					# total combined weighted integration time
					t["tottime_"+str(wl)] = tottime

					# flight numbers
					t["Flight_ID"] = flightstr

				# if none of the measurements are valid
				else:
					t["F"+str(wl)] = 0.0
					t["e_F"+str(wl)] = 0.0
					t.mask["F"+str(wl)] = True
					t.mask["e_F"+str(wl)] = True
					t.mask["flag_F"+str(wl)] = True
		# add fields
		if nFields!=0:
			newRow = hstack([newRow,t])
		#now that this is done for all wavelenths, we need to add it to the table
		if i==0:
			allrows=newRow
		else:
			allrows = vstack([allrows,newRow],join_type='outer')
	return Table(allrows,masked=True)

from astroquery.vizier import Vizier,Conf
import astropy.coordinates as coord
def search_vizier(newPhot,folder_export="/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"):
	'''
	This function searches in a list of Vizier catalogs for fluxes of sources that are within a small radius of each of newPhot's sources.
	'''
	# write here your catalog list
	catlist=['II/328/allwise','J/AJ/144/192/table4','II/332/c2d','II/332/c2d','J/ApJS/184/18/table4',\
			'J/ApJS/184/18/table4','II/125/main','II/298/fis','II/327/ysoc','J/ApJ/638/293/data',\
			'J/ApJ/684/1240/cores','J/A+A/498/167/sources','II/246/out']

	# names associated with the catalogs, needs to be of the same size as the previous list
	namelist=['allwise','megeath','c2d','c2d','guth','guth','iras','akari','akariyso','enoch06','enoch08','van_Kempen','2mass']

	# list of the Vizier names to query and add back into newPhot. We append the catalog name to +_r, _RAJ2000 and _DEJ2000
	# for some strange reason, the amount of fields one can query is limited, so we have to break up the searches
	collist=[['+_r','_RAJ2000','_DEJ2000','W1mag','e_W1mag','W2mag','e_W2mag','W3mag','e_W3mag','W4mag','e_W4mag'],\
		#['Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag'],\
		['+_r','_RAJ2000','_DEJ2000','[3.6]','e_[3.6]','[4.5]','e_[4.5]','[5.8]','e_[5.8]','[8.0]','e_[8.0]','[24]','e_[24]'],\
		#['+_r','_RAJ2000','_DEJ2000','FJ','e_FJ','FH','e_FH','FKs','e_FKs','FIR1','e_FIR1','FIR2','e_FIR2','FIR3','e_FIR3'],\
		['+_r','_RAJ2000','_DEJ2000','FIR1','e_FIR1','FIR2','e_FIR2','FIR3','e_FIR3'],\
		['FIR4','e_FIR4','FMP1','e_FMP1','FMP2','e_FMP2'],\
		['+_r','_RAJ2000','_DEJ2000','Ksmag','e_Ksmag','3.6mag','e_3.6mag','4.5mag','e_4.5mag','5.8mag','e_5.8mag'],\
		['8.0mag','e_8.0mag','24mag','e_24mag'],\
		['+_r','_RAJ2000','_DEJ2000','Fnu_12','e_Fnu_12','Fnu_25','e_Fnu_25','Fnu_60','e_Fnu_60','Fnu_100','e_Fnu_100'],\
		['+_r','_RAJ2000','_DEJ2000','S65','e_S65','S90','e_S90','S140','e_S140','S160','e_S160'],\
		['+_r','_RAJ2000','_DEJ2000','F65','F90','F140','F160'],\
		['+_r','_RAJ2000','_DEJ2000','Fp','e_Fp','AV'],\
		['+_r','_RAJ2000','_DEJ2000','Sp','Tmass'],\
		['+_r','_RAJ2000','_DEJ2000','S70','S450','S850','S1300'],\
		['+_r','_RAJ2000','_DEJ2000','Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag']]

	# timeout setting
	Conf.timeout=200
	
	# only want the closest source, the rest is useless for now
	Conf.row_limit=1

	# format current table for Vizquery input
	newPhot.rename_column('RA','_RAJ2000')
	newPhot.rename_column('DEC','_DEJ2000')
	newPhot['_RAJ2000'].unit = u.deg
	newPhot['_DEJ2000'].unit= u.deg

	# create new table
	newdata = Table(masked=True)

	# sort by sources
	sources = newPhot.group_by('SOFIA_name')
	for source,key in zip(sources.groups,sources.groups.keys):
		RA = source['_RAJ2000'][0]
		DEC = source['_DEJ2000'][0]

		# dist is the search radius in arcsec, different whether it is an extended source or a point source
		if source['Property'][0] == 'Extended':
			dist = 10
		else:
			dist = 3

		# copy the source table
		results=source.copy()

		# loop on catalogs
		for j in range(len(catlist)):
			# setting up search
			catalog=catlist[j]
			name=namelist[j]
			v=Vizier(columns=collist[j])
		
			# only 1 row is desired (might be redundant with Conf.row_limit=1)
			v.ROW_LIMIT=1

			# specifics for certain catalogs
			if 'enoch' in name:
				dist=20
			if 'akari' in name:
				dist=15
			if 'iras' in name:
				dist=20
			if '2mass' in name and source['Property'][0] != 'Extended':
				dist=2
			elif '2mass' in name and source['Property'][0] == 'Extended':
				dist=5
			
			# search command
			result=v.query_region(coord.SkyCoord(ra=RA,dec=DEC,unit=(u.deg,u.deg),frame='fk5'),radius=dist*u.arcsec,catalog=catalog)

			# if there are results, that's good news!
			if len(result.keys())>0:
				if '_RAJ2000' in collist[j]:
					# append _name after those columns
					for col in ['_r','_RAJ2000','_DEJ2000']:
						result[0].rename_column(col,col+"_"+name)
				else:
					# this is those queries for which we don't want the RA and DEC
					result[0].remove_column('_RAJ2000')
					result[0].remove_column('_DEJ2000')

				# stack results
				results=hstack([results,result[0]])

		# add line to new table, by ignoring metadata merging conflicts
		if source['SOFIA_name'][0]==newPhot['SOFIA_name'][0]:
			newdata=results
		else:
			newdata = vstack([newdata,results],metadata_conflicts='silent')
	return newdata	

# this defines the zero points for magnitude to flux conversions
zeroPoints = Table(names=('w1','w2','w3','w4','J','H','K','Ks','irac1','irac2','irac3','irac4','mips1','mips2'))
zeroPoints.add_row((306.681,170.663,29.0448,8.2839,1594,1024,676,666.7,280.9,179.7,115,64.9,7.17,0.0778)) # zero points in Jy


def mag2flux(allNewPhot,fname,newfname,refname):
	'''
	This converts magnitudes and magnitude errors to fluxes and errors in Jy
	'''
	# small bug from Vizier to deal with
	if fname.startswith('_'):
		errorname = 'e'+fname
	else:
		errorname='e_'+fname
	#allNewPhot['e_'+newfname] = zeroPoints[refname][0]*pow(10,-0.4*(allNewPhot[fname]-allNewPhot[errorname]))
	
	# convert the magnitude to a flux
	allNewPhot[newfname] = zeroPoints[refname][0]*pow(10,-0.4*allNewPhot[fname])

	# deal with the 2MASS fluxes, that give no errors when the fluxes are upper limits
	if fname in ['Jmag','Hmag', 'Kmag']:

		# when error is masked (means upper limit), then error is just as much as the flux
		allNewPhot[errorname][allNewPhot.mask[errorname]] = allNewPhot[newfname][allNewPhot.mask[errorname]]

		# copy masks over to make sure that errors are unmasked now
		allNewPhot.mask[errorname] = allNewPhot.mask[newfname]

		# Put all flags to nominal
		allNewPhot['flag_'+newfname] = 'N'

		# except the ones that are upper limits
		allNewPhot['flag_'+newfname][allNewPhot.mask[errorname]] = 'U'

		# rename column
		allNewPhot['e_'+newfname] =  allNewPhot[errorname]
		#print allNewPhot
	else:

		# if error is present in magnitude, then use aproximation of error in Jy, error_Jy ~ flux * error_mag
		allNewPhot['e_'+newfname] = allNewPhot[newfname] * allNewPhot[errorname]
	


def convert_table(table):
	'''
	Cleans up the photometry table, and gives priorities to fluxes
	'''
	allNewPhot = table.copy()

	# convert all of these fields to fluxes
	mag2flux(allNewPhot,'W1mag','w1','w1')
	mag2flux(allNewPhot,'W2mag','w2','w2')
	mag2flux(allNewPhot,'W3mag','w3','w3')
	mag2flux(allNewPhot,'W4mag','w4','w4')
	mag2flux(allNewPhot,'Jmag','j','J')
	mag2flux(allNewPhot,'Hmag','h','H')
	mag2flux(allNewPhot,'Kmag','ks','Ks')
	mag2flux(allNewPhot,'__3.6_','i1_megeath','irac1')
	mag2flux(allNewPhot,'__4.5_','i2_megeath','irac2')
	mag2flux(allNewPhot,'__5.8_','i3_megeath','irac3')
	mag2flux(allNewPhot,'__8.0_','i4_megeath','irac4')
	mag2flux(allNewPhot,'__24_','m1_megeath','mips1')
	mag2flux(allNewPhot,'_3.6mag','i1_guth','irac1')
	mag2flux(allNewPhot,'_4.5mag','i2_guth','irac2')
	mag2flux(allNewPhot,'_5.8mag','i3_guth','irac3')
	mag2flux(allNewPhot,'_8.0mag','i4_guth','irac4')
	mag2flux(allNewPhot,'_24mag','m1_guth','mips1')

	# set 10% errors for those fluxes which do not have errors provided
	for col in ['S70','S450','S850','S1300','Sp']:
		allNewPhot['e_'+col] = 0.1*allNewPhot[col]
	
	# units and masking
	for col in ['w1','w2','w3','w4','j','h','ks','FIR1','FIR2','FIR3','FIR4','FMP1','FMP2',\
		'i1_megeath','i2_megeath','i3_megeath','i4_megeath','m1_megeath','i1_guth','i2_guth','i3_guth','i4_guth','m1_guth','Fp','Sp']:
		#allNewPhot[col] /= 1000.
		#allNewPhot['e_'+col] /= 1000.
		allNewPhot[col].unit = u.Jy
		allNewPhot['e_'+col].unit = u.Jy
		for j in range(len(allNewPhot)):
			if np.isnan(allNewPhot[col][j]) or allNewPhot[col][j] <= 0.: allNewPhot.mask[col][j] =True
			if np.isnan(allNewPhot['e_'+col][j]) or allNewPhot['e_'+col][j] <= 0.: allNewPhot.mask['e_'+col][j] =True

	# some of those fluxes are given in mJy instead of Jy
	for col in ['FIR1','FIR2','FIR3','FIR4','FMP1','FMP2','S450','S850','S1300']:
		allNewPhot[col] /= 1000.
		allNewPhot['e_'+col] /= 1000.

	# update column names			
	allNewPhot.rename_column('FIR1','i1')
	allNewPhot.rename_column('FIR2','i2')
	allNewPhot.rename_column('FIR3','i3')
	allNewPhot.rename_column('FIR4','i4')
	allNewPhot.rename_column('FMP1','m1')
	allNewPhot.rename_column('FMP2','m2')
	allNewPhot.rename_column('e_FIR1','e_i1')
	allNewPhot.rename_column('e_FIR2','e_i2')
	allNewPhot.rename_column('e_FIR3','e_i3')
	allNewPhot.rename_column('e_FIR4','e_i4')
	allNewPhot.rename_column('e_FMP1','e_m1')
	allNewPhot.rename_column('e_FMP2','e_m2')

	# Sets flux priorities:
	# c2d > gutermuth and megeath
	for i in range(len(allNewPhot)):
		for val in ['i1','i2','i3','i4','m1']:			
			if allNewPhot.mask[val][i] == True:
				### if the gutermuth values are unmasked, use them!
				if allNewPhot.mask[val+'_guth'][i] == False: 
					print 'replaced',allNewPhot[val][i],allNewPhot.mask[val][i],' by ',allNewPhot[val+'_guth'][i],allNewPhot.mask[val+'_guth'][i]
					allNewPhot[val][i] = allNewPhot[val+'_guth'][i]
					allNewPhot['e_'+val][i] = allNewPhot['e_'+val+'_guth'][i]
				elif allNewPhot.mask[val+'_megeath'][i] == False: 
					print 'replaced',allNewPhot[val][i],allNewPhot.mask[val][i],' by ',allNewPhot[val+'_megeath'][i],allNewPhot.mask[val+'_megeath'][i]
					allNewPhot[val][i] = allNewPhot[val+'_megeath'][i]
					allNewPhot['e_'+val][i] = allNewPhot['e_'+val+'_megeath'][i]
		for val in ['m2']:			
			if allNewPhot.mask[val][i] == True:
				### if the gutermuth values are unmasked, use them!
				if allNewPhot.mask['S70'][i] == False: 
					print 'replaced',allNewPhot[val][i],allNewPhot.mask[val][i],' by ',allNewPhot['S70'][i],allNewPhot.mask['S70'][i]
					allNewPhot[val][i] = allNewPhot['S70'][i]
					allNewPhot['e_'+val][i] = allNewPhot['e_S70'][i]
		for val in ['Fp']:			
			if allNewPhot.mask[val][i] == True:
				if allNewPhot.mask['Sp'][i] == False: 
					print 'replaced',allNewPhot[val][i],allNewPhot.mask[val][i],' by ',allNewPhot['Sp'][i],allNewPhot.mask['Sp'][i]
					allNewPhot[val][i] = allNewPhot['Sp'][i]
					allNewPhot['e_'+val][i] = allNewPhot['e_Sp'][i]

	# errors for IRAS are given in % of the source flux
	for col in ['Fnu_12','Fnu_25','Fnu_60','Fnu_100']:
		allNewPhot['e_'+col] = allNewPhot[col]*allNewPhot['e_'+col]/100. 
		allNewPhot['e_'+col].unit = u.Jy

	# finally, rename the RA and DEC columns to things that are used in the rest of the code
	allNewPhot.rename_column('_RAJ2000','RA')
	allNewPhot.rename_column('_DEJ2000','DEC')
	
	return allNewPhot


def addHerschel(sourcetable,Herschdir="/cardini3/mrizzo/2012SOFIA/Herschel_Mosaics/",\
	Hersch_list = ['Oph-70microns.fits','Oph-160microns.fits','Oph-250microns.fits','Oph-350microns.fits','Oph-500microns.fits']):
	'''
	this function adds the photometry from the Herschel maps
	it uses aperture photometry, and a local background.
	'''
	sources = sourcetable.group_by('SOFIA_name')
	sourcelist = []

	# loop on sources
	for key,source in zip(sources.groups.keys,sources.groups):
		
		# select only Ophiuchus sources
		if 'Oph' in source['SOFIA_name'][0]:
			RA = source['RA'][0]
			DEC = source['DEC'][0]

			# look at all the files for a match
			for f in [filename for filename in Hersch_list]:

				# open image
				hdu = fits.open(Herschdir+f)
				prihdr = hdu[1].header
				w = wcs.WCS(hdu[1].header)
				img = hdu[1].data

				# find pixel scale in asec
				pix_scale = prihdr['CDELT2']*3600

				# Masking
				image = np.ma.masked_invalid(img)
				mask = np.isnan(image) | image.mask

				# Converts RA and DEC to pixel coordinates
				X,Y = w.wcs_world2pix(RA,DEC,1)

				# For each band, set the aperture and annulus according to guidelines
				if '70' in f:
					#r_out=12./(pix_scale)
					#annu_in=35./pix_scale
					#annu_out=45./pix_scale
					#aper_corr=1./0.802
					r_out=6./(pix_scale)
					annu_in=25./pix_scale
					annu_out=35./pix_scale
					aper_corr=1.5711
					if 'Oph.4' in source['SOFIA_name'][0]:
						r_out=20./(pix_scale)
						aper_corr=1.
					elif 'Oph.11' in source['SOFIA_name'][0]:
						r_out=30./(pix_scale)
						aper_corr=1.
					name = 'H70'
					bkgd_reg=phot.CircularAnnulus((X,Y),annu_in,annu_out)
				if '160' in f:
					r_out=12./(pix_scale)
					annu_in=25./pix_scale
					annu_out=35./pix_scale
					aper_corr=1.4850
					#r_out=22./(pix_scale)
					#annu_in=35./pix_scale
					#annu_out=45./pix_scale
					#aper_corr=1./0.817
					if 'Oph.4' in source['SOFIA_name'][0]:
						r_out=30./(pix_scale)
						aper_corr=1.
					elif 'Oph.11' in source['SOFIA_name'][0]:
						r_out=30./(pix_scale)
						aper_corr=1.
					name = 'H160'
				if '250' in f:
					r_out=22./(pix_scale)
					annu_in=60./pix_scale
					annu_out=90./pix_scale
					aper_corr=1.2697
					name = 'H250'
				if '350' in f:
					r_out=30./(pix_scale)
					annu_in=60./pix_scale
					annu_out=90./pix_scale
					aper_corr=1.2271
					name = 'H350'
				if '500' in f:
					r_out=42./(pix_scale)
					annu_in=60./pix_scale
					annu_out=90./pix_scale
					aper_corr=1.2194
					name = 'H500'
				if '250' in f:
					image *= pix_scale**2/469.7
				if '350' in f:
					image *= pix_scale**2/831.7
				if '500' in f:
					image *= pix_scale**2/1793.5

				# check is source is in the image
				if (X>r_out and X<image.shape[1]-r_out) and (Y>r_out and Y<image.shape[0]-r_out) \
					and not image.mask[Y-r_out:Y+r_out,X-r_out:X+r_out].any():

					# set up aperture and calculate flux
					ap = phot.CircularAperture((X,Y),r=r_out)
					flux = phot.aperture_photometry(image*~mask,ap,mask = mask)

					# set up annulus and calculate background value and rms
					bkg_ap = phot.CircularAnnulus((X,Y),annu_in,annu_out)
					bkg, bkg_area = calc_masked_aperture(bkg_ap, image*~mask, method='mmm', mask=mask)
					bkg_rms, bkg_aps = calc_bkg_rms(bkg_ap, image*~mask, src_ap_area = ap.area(),rpsrc = r_out, mask=mask, min_ap=6)

					# correct for background
					photometry=(flux["aperture_sum"]  - bkg*np.pi*r_out**2)* aper_corr
					source[name] = photometry

					# calculate error
					source['e_'+name] = bkg_rms*aper_corr

					# if photometry is smaller than error, then set the photometry to the error and raise a flag
					if source[name][0] < source['e_'+name][0]:
						source[name][0] = source['e_'+name][0]
						source['flag_'+name] = 'U'
					else:
						source['flag_'+name] = 'N'
		sourcelist.append(source)
	return vstack(sourcelist)

def calcAlpha(sourcetable,columnlist,errorlist,wavelength):
	'''
	Calculates the spectral index scross the columns
	'''
	newtable = sourcetable.copy()

	# add spectral columns
	newtable.add_column(Column(np.zeros(len(sourcetable)),name="alpha"))
	newtable.add_column(Column(np.zeros(len(sourcetable)),name="e_alpha"))

	# loop on sources
	for i in range(len(sourcetable)):

		# load info about source
		sourceID = sourcetable['SOFIA_name'][i]
		RA = sourcetable['RA'][i]
		DEC = sourcetable['DEC'][i]
		source = sourcetable[columnlist][i]

		# convert flux list to numpy masked array
		ntable = nptable(source)

		# for each flux value, force the mask if flux==0
		for j in range(len(ntable)):
			if ntable[j]==0:
				ntable.mask[j]=True

		# creates the array of wavelengths where the fluxes are masked
		mwl = np.ma.masked_where(ntable.mask,wavelength)

		# calculate lambda*Flambda
		ntable *=mwl

		# log(lambdaFlambda)
		logntable = np.log10(ntable)

		# don't do anything if all fluxes are zero
		if np.ma.count(ntable)==0:
			print " All values masked for source %s - skipping" % (sourceID)
		else:
			# calculate error table
			errtottable = sourcetable[errorlist][i]
			errtable = np.log10(nptable(errtottable)*mwl)
	
			# fit results to a line in loglog space
			fitresults,cov =  np.polyfit(mwl,logntable,1,w=errtable,cov=True)

			# populate table
			newtable["alpha"][i] = fitresults[0]
			newtable['e_alpha'][i] = np.sqrt(cov[0,0])
	return newtable

def addFitResults(sourcetable,folder_export=""):
	'''
	Adds some fit results to the table
	'''
	newtable = sourcetable
	newtable.add_column(Column(np.zeros(len(sourcetable)),name="chi2"))
	newtable.add_column(Column(np.zeros(len(sourcetable)),name="ltot"))
	# add any other parameter you want to add, and add it below as well

	
	for i in range(len(sourcetable)):
		if sourcetable['Property'][i] == 'Isolated' or sourcetable['Property'][i] == 'Clustered':
			sourceID = sourcetable['SOFIA_name'][i]

			# find correct output file
			fname = folder_export+"plot_"+sourcetable['Cluster'][i]+'/'+sourceID+'.tab'

			# load and pick values
			if os.path.isfile(fname):
				fitparams = pickle.load(open(fname,'r'))
				newtable["chi2"][i] = fitparams['chi2'][0]
				newtable["ltot"][i] = fitparams['ltot'][0]

	return newtable

def do_IRAC_phot(table,cluster,r_out=4,r_annulus=6,IRAC_IMG_FOLDER = '/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/'):
	'''
	This replaces the IRAC photometry from the sources from 'cluster' in 'sourcetable', and does the photometry manually
	IRAC images need to be present and contain the cluster name and the string 'IRAC.X', where X is the band.
	Needs to manually input the IRAC aperture correction for each band
	r_out=4  corresponds to an aperture of radius 2.4" on the sky

	'''
	# create IRAC aperture corrections table
	# from http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/27/
	aper_corr = Table(np.array([1.208,1.220,1.349,1.554]),names=('1','2','3','4'))

	# copy input table
	sourcetable = table.copy()

	# check if flag columns exist; if not, create flag columns for IRAC sources
	for band in ['1','2','3','4']:
		if 'flag_i'+band not in sourcetable.columns: 
			sourcetable['flag_i'+band] = 'N'

	# read input cluster
	for i in range(len(sourcetable)):
		
		# test on the cluster
		if sourcetable['Cluster'][i] == cluster:
			
			# determine source location
			RA = sourcetable['RA'][i];DEC = sourcetable['DEC'][i]
		
			# loop on the IRAC bands
			for band in ['1','2','3','4']:
			
				# find corresponding IRAC image
				dirlist = os.listdir(IRAC_IMG_FOLDER)
				filelist = [f for f in dirlist if 'IRAC.'+band in f if cluster in f]

				# tests
				if len(filelist) != 1:
					print 'File not found, or too many files founts!'
				else:
					iracimg = filelist[0]
				
				# Load image & X,Y coordinates of source
				hdu = fits.open(IRAC_IMG_FOLDER+iracimg)
				w = wcs.WCS(hdu[0].header)
				img = hdu[0].data
				image = np.ma.masked_invalid(img)
				mask = np.isnan(image) | image.mask
				X,Y = w.wcs_world2pix(RA,DEC,1)

				# Convert MJy/sr to Jy/pix
				sr_per_pix = (0.6*4.848e-6)**2
				image *= 1e6*sr_per_pix

				# define apertures using use-defined annulus size
				ap = phot.CircularAperture((X,Y),r=r_out)
				bkg_ap = phot.CircularAnnulus((X,Y),r_annulus,r_annulus+r_out)
			
				# Photometry
				flux = phot.aperture_photometry(image*~mask,ap,mask = mask)
				bkg, bkg_area = calc_masked_aperture(bkg_ap, image*~mask, method='mmm', mask=mask)
				bkg_rms, bkg_aps = calc_bkg_rms(bkg_ap, image*~mask,src_ap_area = np.pi*r_out**2, rpsrc = r_out, mask=mask, min_ap=6)

				# replace fluxes in table, using correct aperture correction
				sourcetable["i"+band][i] = (flux["aperture_sum"][0] - bkg*ap.area())*aper_corr[band]

				# calculate errors
				sourcetable["e_i"+band][i] = bkg_rms*aper_corr[band]

				# flag/modify data if flux is too low
				if sourcetable["i"+band][i] < sourcetable["e_i"+band][i]:
					sourcetable["i"+band][i] = sourcetable["e_i"+band][i]
					sourcetable["flag_i"+band][i] = 'U'

	# when photometry is done, return modified table
	return sourcetable


def do_manual_2mass_phot(table):
	'''
	This function just adjusts the table for IRAS20050, through visual inspection of the 2MASS images
	'''
	sourcetable = table.copy()

	# approximate sensitivity limits from 2MASS (VI.1. Analysis of the Release Catalogs)	
	Jlim = 0.0007
	Hlim = 0.0009
	Klim = 0.0015
	limits = Table(np.array([Jlim,Hlim,Klim]),names=('j','h','k'))

	# read input cluster
	for i in range(len(sourcetable)):
		
		# test on the cluster
		if sourcetable['Cluster'][i] == 'IRAS20050':

			# source number 2 has no matching believable flux in 2MASS
			# source 4 likely has no believable flux either
			if "2" or "4" in sourcetable['SOFIA_name'][i]:
				for band in ['j','h','k']:
					sourcetable[band][i] = limits[band];sourcetable.mask[band][i] = False
					sourcetable['e_'+band][i] = limits[band];sourcetable.mask['e_'+band][i] = False
					sourcetable['flag_'+band][i] = 'U';sourcetable.mask['flag_'+band][i] = False

	return sourcetable


def nptable(sourcetable,columnlist=None,minmask = None):
	'''
	This function converts a table to a masked numpy array
	'''
	# simply return a numpy masked array
	if columnlist==None:
		tarray = np.ma.array([sourcetable[col] for col in sourcetable.columns])
	else:
		tarray = np.ma.array([sourcetable[col] for col in sourcetable[columnlist].columns])
	
	# use this to clip the array to some minimum value; optional
	if minmask!=None:
		tarray[tarray<minmask] = np.ma.masked
	return tarray.T

# imports relating to the sedfitter package from T. Robitaille
import gzip
import astropy.constants as const
from sedfitter import sed
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker
from sedfitter import plot_source_data
from sedfitter import six
from sedfitter.extinction import Extinction
from sedfitter.fit_info import FitInfo, FitInfoFile
from sedfitter.sed import SED
from sedfitter.sed.cube import SEDCube
from sedfitter.utils import io
from sedfitter.utils import parfile
from sedfitter.utils.formatter import LogFormatterMathtextAuto

# a kiloparsec
KPC = 3.086e21

# color settings
color = {}
color['gray'] = '0.75'
color['black'] = '0.00'
color['full'] = []
color['full'].append((0.65, 0.00, 0.00))
color['full'].append((0.20, 0.30, 0.80))
color['full'].append((1.00, 0.30, 0.30))
color['full'].append((1.00, 0.30, 0.60))
color['full'].append((0.30, 0.80, 0.30))
color['full'].append((0.50, 0.10, 0.80))
color['full'].append((0.20, 0.60, 0.80))
color['full'].append((1.00, 0.00, 0.00))
color['full'].append((0.50, 0.25, 0.00))
color['full'].append((0.90, 0.90, 0.00))
color['full'].append((0.00, 0.50, 0.00))
color['faded'] = []
color['faded'].append((1.00, 0.70, 0.70))
color['faded'].append((0.70, 0.70, 0.80))
color['faded'].append((1.00, 0.80, 0.70))
color['faded'].append((1.00, 0.75, 1.00))
color['faded'].append((0.70, 0.80, 0.70))
color['faded'].append((0.75, 0.60, 0.80))
color['faded'].append((0.70, 0.75, 0.80))
color['faded'].append((1.00, 0.70, 0.80))
color['faded'].append((0.90, 0.80, 0.70))
color['faded'].append((0.90, 0.90, 0.70))
color['faded'].append((0.50, 0.90, 0.50))
fp = FontProperties(size='x-small')

# heritage function, modified by us
def tex_friendly(string):
    return string
    # return string.replace('_', '\_').replace('%', '\%')


def plot_source_info(ax, i, info, plot_name, plot_info):
    '''
    This function is found in the sedfitter source code, with minimal modifications
    '''
    labels = []

    if plot_name:
        labels.append(tex_friendly(info.source.name))

    if plot_info:
        labels.append("Model: %s" % tex_friendly(info.model_name[i]))
        if i == 0:
            labels.append("Best fit")
        else:
            labels.append("Fit: %i" % (i + 1))
        labels.append("$\chi^2$ = %10.3f    A$_{\\rm V}$ = %5.1f    Scale = %5.2f" % (info.chi2[i], info.av[i], info.sc[i]))

    pos = 0.95
    for label in labels:
        ax.text(0.01, pos, label, horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                fontproperties=fp)
        pos -= 0.06


def plotsedfit(input_fits, output_ax, select_format=("N", 1), plot_max=None,
         plot_mode="A", sed_type="interp", show_sed=True, show_convolved=False,
         x_mode='A', y_mode='A', x_range=(1., 1.), y_range=(1., 2.),
         plot_name=False, plot_info=True, format='pdf', sources=None, memmap=True,av=None):
    """
    Make SED plots on top of already existing axes
    Modified from original plot.py in sedfitter module

    Parameters
    ----------
    input_fits : str or :class:`sedfitter.fit_info.FitInfo` or iterable
        This should be either a file containing the fit information, a
        :class:`sedfitter.fit_info.FitInfo` instance, or an iterable containing
        :class:`sedfitter.fit_info.FitInfo` instances.
    output_dir : str, optional
        If specified, plots are written to that directory
    select_format : tuple, optional
        Tuple specifying which fits should be plotted. See the documentation
        for a description of the tuple syntax.
    plot_max : int, optional
        Maximum number of fits to plot
    plot_mode : str, optional
        Whether to plot all fits in a single plot ('A') or one fit per plot
        ('I')
    sed_type : str, optional
        Which SED should be shown:
            * `largest`: show the SED for the largest aperture specified.
            * `smallest`: show the SED for the smallest aperture specified.
            * `largest+smallest`: show the SEDs for the largest and smallest
              apertures specified.
            * `all`: show the SEDs for all apertures specified.
            * `interp`: interpolate the SEDs to the correct aperture at each
              wavelength (and interpolated apertures in between), so that a
              single composite SED is shown.
    show_sed : bool, optional
        Show the SEDs
    show_convolved : bool, optional
        Show convolved model fluxes
    x_mode : str, optional
        Whether to automatically select the wavelength range ('A'), or whether
        to use manually set values ('M').
    x_range : tuple, optional
        If x_mode is set to 'M', this is the range of wavelengths to show. If
        x_mode is set to 'A', this is the marging to add around the wavelength
        range (in dex).
    y_mode : str, optional
        Whether to automatically select the flux range ('A'), or whether to
        use manually set values ('M').
    y_range : tuple, optional
        If y_mode is set to 'M', this is the range of fluxes to show. If
        y_mode is set to 'A', this is the marging to add around the flux
        range (in dex).
    plot_name : bool, optional
        Whether to show the source name on the plot(s).
    plot_info : bool, optional
        Whether to show the fit information on the plot(s).
    format : str, optional
        The file format to use for the plot, if output_dir is specified.
    sources : list, optional
        If specified, gives the list of sources to plot. If not set, all
        sources will be plotted
    """
    # feed in infor from previous fit
    fin = FitInfoFile(input_fits, 'r')

    # define apertures and wavelengths
    wav = np.array([f['wav'].to(u.micron).value for f in fin.meta.filters])
    ap = np.array([f['aperture_arcsec'] for f in fin.meta.filters])
    unique_ap = np.unique(ap)
    unique_ap.sort()

    # Read in model parameters
    modpar = parfile.read("%s/models.conf" % fin.meta.model_dir, 'conf')
    if not 'version' in modpar:
        modpar['version'] = 1

    model_dir = None
    sed_cube = None

    # loop on source
    for info in fin:

        if sources is not None and info.source.name not in sources:
            continue

        if modpar['version'] == 2 and model_dir != info.meta.model_dir:
            sed_cube = SEDCube.read(os.path.join(info.meta.model_dir, 'flux.fits'), memmap=memmap)
            model_dir = info.meta.model_dir

        # Filter fits
        info.keep(select_format[0], select_format[1])

        if plot_max:
            info.keep('N', plot_max)

        if show_convolved and info.model_fluxes is None:
            raise Exception("Cannot plot convolved fluxes as these are not included in the input file")

        if info.n_fits == 0 and output_dir is None:
            figures[info.source.name] = {'source': info.source, 'filters': info.meta.filters}

	# loop on fits that match criterion
        for i in range(info.n_fits - 1, -1, -1):

            # Initalize lines and colors list
            if (plot_mode == 'A' and i == info.n_fits - 1) or plot_mode == 'I':
                lines = []
                colors = []
                if show_convolved:
                    conv = []

            if (plot_mode == 'A' and i == 0) or plot_mode == 'I':
                if sed_type in ['interp', 'largest']:
                    color_type = 'black'
                else:
                    color_type = 'full'
            else:
                if sed_type in ['interp', 'largest']:
                    color_type = 'gray'
                else:
                    color_type = 'faded'

	    # deals with previous versions
            if modpar['version'] == 1:
                if modpar['length_subdir'] == 0:
                    s = SED.read(info.meta.model_dir + '/seds/' + info.model_name[i] + '_sed.fits')
                else:
                    s = SED.read(info.meta.model_dir + '/seds/%s/%s_sed.fits' % (info.model_name[i][:modpar['length_subdir']], info.model_name[i]))
            elif modpar['version'] == 2:
                s = sed_cube.get_sed(info.model_name[i])

            # Convert to ergs/cm^2/s
            #s.flux = s.flux.to(u.erg / u.cm**2 / u.s, equivalencies=u.spectral_density(s.nu))

            s = s.scale_to_distance(10. ** info.sc[i] * KPC)
	    if av is None:
            	s = s.scale_to_av(info.av[i], info.meta.extinction_law.get_av)

            if sed_type == 'interp':
                apertures = ap * 10. ** info.sc[i] * 1000.
                flux = s.interpolate_variable(wav, apertures)
            elif sed_type == 'largest':
                apertures = np.array([ap.max()]) * 10. ** info.sc[i] * 1000.
                flux = s.interpolate(apertures)
            elif sed_type == 'largest+smallest':
                apertures = np.array([ap.min(), ap.max()]) * 10. ** info.sc[i] * 1000.
                flux = s.interpolate(apertures)
            elif sed_type == 'all':
                apertures = unique_ap * 10. ** info.sc[i] * 1000.
                flux = s.interpolate(apertures)

            if flux.ndim > 1:
                for j in range(flux.shape[1]):
                    lines.append(np.column_stack([s.wav, flux[:, j]]))
                    colors.append(color[color_type][j])
            else:
                lines.append(np.column_stack([s.wav, flux]))
                colors.append(color[color_type])

            if show_convolved:
                conv.append(10. ** (info.model_fluxes[i, :] - 26. + np.log10(3.e8 / (wav * 1.e-6))))

	    # set the output axes
            ax = output_ax

	    # plotting in dedicated directory; thick line for the best fit, and gray lines for the other fits
            if (plot_mode == 'A' and i == 0) or plot_mode == 'I':

                    if show_sed:
                        ax.add_collection(LineCollection(lines, colors=colors))

                    if show_convolved:
                        for j in range(len(conv)):
                            ax.plot(wav, conv[j], color=colors[j], linestyle='solid', marker='o', markerfacecolor='none', markeredgecolor=colors[j])

                   if plot_mode == 'A':
                        ax = plot_source_info(ax, 0, info, plot_name, plot_info)
                    else:
                        ax = plot_source_info(ax, i, info, plot_name, plot_info)

    # close file and end script
    fin.close()

# these set the names of the variables in the model fitting; each of these parameters can be individually retrieved from the fits
name_col = ['fit_id', 'model_name', 'chi2', 'av', 'scale', 'time', 'massc', 'rstar', 'tstar', 'mdot',
		'rmax', 'theta', 'rmine', 'mdisk', 'rmaxd', 'rmind', 'rmind(au)', 'rc', 'rchole', 'zmin',
		'a', 'b', 'alpha', 'rhoconst', 'rhoamb', 'mdotdisk', 'incl.', 'av_int', 'tau60', 'ltot', 'h100']
latex_col = ['$fit_id$','$model_name$',r'$\chi_{best}^2$',r'$A_V$','scale','time',r'$M_c$',r'$R_\star$',r'$T_\star$',r'$\dot{M}$',
		r'$R_{max}$', r'$\theta$', r'$R_{env}^{min}$',r'$M_{disk}$', r'$R_{disk}^{max}$', r'$R_{disk}^{min}$',
		r'$R_{disk}^{min,AU}$', r'$R_c$', r'$R_{c,hole}$', r'$Z_{min}$', 'a', 'b', r'$\alpha$', r'$\rho_{const}$',
		r'$\rho_{amb}$', r'$\dot{M}_{disk}$', r'$i$',r'$A_V^{int}$', r'$\tau_{60}$', r'$L_{tot}$', r'$h_{100}$']
nametable = Table(names=name_col, dtype = ['<S30' for col in name_col])
nametable.add_row(latex_col)

def newdet_filename(cluster,n,spitzerfolder="/n/a2/mrizzo/Dropbox/SOFIA/Spitzer_Mosaics/",sofiafolder="/n/a2/mrizzo/Dropbox/SOFIA/Mosaics/"):
	'''
	This function deermines the name of the .fits file where the SOFIA data is for a corresponding cluster and n
	'''
	if cluster=="Oph":
		clustervar="Ophiuchus"
	else:
		clustervar=cluster
	if n==3:
		return sofiafolder+cluster+".11.fits"
	elif n==4:
		return sofiafolder+cluster+".19.fits"
	elif n==5:
		return sofiafolder+cluster+".31.fits"
	elif n==6 or n==7:
		return sofiafolder+cluster+".37.fits"

def get_img_cal(flight_ID,size,folder_export="/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/",calfolder="/cardini3/mrizzo/2012SOFIA/2014/Calibrators/"):
	'''
	This function returns the image from the calibrator corresponding to the flight when the 37um data was taken
	'''

	caltable = pickle.load(open(folder_export+"caltable.data","r"))

	# group calibrators by flight number
	flights = caltable.group_by('Flight_ID')

	# init image
	image = np.zeros((2*size,2*size))

	# loop on flight number
	for key,group in izip(flights.groups.keys,flights.groups):
		if flight_ID in key['Flight_ID']:
			wavelengths = group.group_by('Wavelength')
			for wlkey,wlgroup in izip(wavelengths.groups.keys,wavelengths.groups):
				if 37.1 == wlkey['Wavelength']:
					# sort from dimmest to brightest
					wlgroup.sort('Phot_val')

					# pick brightest calibrator!
					filepath = calfolder+"sofia_"+wlgroup['Flight_ID'][-1]+"/"+wlgroup['Filename'][-1]

					# open fits file, get centroid and extract image
					hdu = fits.open(filepath)
					img = hdu[0].data
					prihdr = hdu[0].header
					X = prihdr['ST0X']
					Y = prihdr['ST0Y']
					if len(img.shape)==3:
						img = img[0,:,:]
					image = img[Y-size:Y+size,X-size:X+size]
	return image

# this picks the color palette
colors = sns.color_palette('hls',9)

def newimgcutouts(ax,source,n,sizeasec=10,folder="/cardini3/mrizzo/2012SOFIA/Spitzer_Mosaics/" ):
	'''
	Returns cutouts of images at certain wavlengths
	'''
	# read in info from input
	RA = source['RA'][0]
	DEC = source['DEC'][0]
	prop = source['Property'][0]
	cluster = source['Cluster'][0]
	border = 10

	# set size of snapshot, depending on the source type
	if "Isolated" in prop or "Clustered" in prop:
		sizea = sizeasec
	else:
		sizea = np.maximum(np.maximum(source['SemiMajorAxis'][0],source['SemiMinorAxis'][0])*1.4,sizeasec)
	
	# 1,2 are the IRAC 1 and 4 bands, respectively
	if n in (1,2):
		if n==1:
			num = "1"
		elif n==2:
			num = "4"

		# divide by plate scale
		size = sizea/0.6 

		# init snapshot subimage
		subimg = np.zeros((size,size))

		# loop on files in the image folder
		dirlist = os.listdir(folder)
		for f in dirlist:
			if cluster in f and ("."+num+".") in f:

				# open fits files and test if image is present
				hdu = fits.open(folder+f)
				w = wcs.WCS(hdu[0].header)
				img = hdu[0].data
				image = np.ma.masked_invalid(img)
				X,Y = w.wcs_world2pix(RA,DEC,1)

				# some init for the 31um band to overplot the location of centroids for other data
				if n==5:
					Xmid = X;Ymid=Y
				if (X>border and X<image.shape[1]-border) and (Y>border and Y<image.shape[0]-border) \
					and not image.mask[Y-border:Y+border,X-border:X+border].all():
					print "Found match: "+f
					subimg = image[Y-size:Y+size,X-size:X+size]
					break
				else:
					print "source "+source['SOFIA_name'][0]+" not found in "+f
		if subimg.all() == 0:
			print "Source "+source['SOFIA_name'][0]+" not found in any field"
	else:
		# other than Spitzer images, the rest is SOFIA
		filename = newdet_filename(cluster,n)
		hdu = fits.open(filename)
		w = wcs.WCS(hdu[0].header)
		img = hdu[0].data

		# different plate scale
		size = sizea/0.768
		
		# n=7 is the calibrator at 37um, make use ofprevious functions to find it
		if n==7:
			prihdr = hdu[0].header
			flight_ID = get_from_header(prihdr,"MISSN-ID")
			subimg = get_img_cal(flight_ID,size)
		else:
			image = np.ma.masked_invalid(img)
			# Converts RA and DEC to pixel coordinates and extract image around X,Y
			X,Y = w.wcs_world2pix(RA,DEC,1)
			subimg = image[Y-size:Y+size,X-size:X+size]
		if n==5:
			Xmid = X;Ymid=Y
	
	# plot image within subregion
	ax.imshow(-subimg,origin='lower',interpolation='nearest',cmap=plt.get_cmap('gray'))
	
	# the following overplots a circle at the location of another data catalog RA and DEC for that same source
	# this helps us know if the emission is not colocated
	if n==5:
		namelist=['allwise','megeath','c2d','guth','iras','akari','akariyso','enoch06','enoch08','van_Kempen','2mass]
		colorlist=[colors[2],colors[1],colors[1],colors[1],colors[4],colors[5],colors[5],colors[6],colors[6],colors[8],colors[0]]

		# loop on each catalog
		for i in range(len(namelist)):
			name=namelist[i]
			color=colorlist[i]
		
			# make sure that we did find a matching source in that catalog
			if source.mask['_RAJ2000_'+name][0]==False:
				RAi = source['_RAJ2000_'+name][0]
				DECi = source['_DEJ2000_'+name][0]
				X,Y = w.wcs_world2pix(RAi,DECi,1)
				Xc = X - Xmid+size
				Yc = Y - Ymid+size
				ax.add_patch(plt.Circle((Yc,Xc),radius=2,fc='None',ec=color))

	return subimg,2*sizea

def newdet_label(i):
	'''
	Determines the labeling underneath the snapshot
	'''
	wl = [3.6,8.,11.1,19.7,31.5,37.1]
	microns = "$\mu$m"
	WISE = "IRAC "
	SOFIA = "SOFIA "
	if i in (1,2):
		return WISE+str(wl[i-1])+microns
	elif i == 7:
		return "Calib. 37.1"+microns
	else:
		return SOFIA+str(wl[i-1])+microns


class plots(object):
	'''
	simple class used for plotting different types of points
	'''
	def __init__(self,bands,wl,color,marker,label):
		self.bands=bands
		self.wl = wl
		self.color = color
		self.marker = marker
		self.label = label


def plotData(ax,sourcetable,plots,alpha):
	'''
	plots a set of points with errorbars
	'''
	marker = plots.marker
	columnlist = plots.bands
	errorlist = ['e_'+col for col in plots.bands]
	wllist = plots.wl
	color = plots.color
	label = plots.label
	ax.errorbar(wllist,nptable(sourcetable[columnlist][0])*1e-17*const.c.value/wllist,nptable(sourcetable[errorlist][0])*1e-17*const.c.value/wllist, 
		c=color,alpha=alpha,linestyle="None",marker=marker,label = label,markersize=10)



def markerPlotSED(sourcetable,error=None,show=True,alpha=0.8,folder_export="",\
		linestyle='None',RAstr="RA_ave",DECstr="DEC_ave",cluster=None):
	'''
	this function creates images with SED plots, SED fit results, and snapshots
	'''
	
	# the following block sets the marker, colors, labels, and selects which
	# column of the sourcetable corresponds to which color/marker/labels
	markers = ['v','p','D','^','h','o','*','>','<']
	TwoMASS = plots(['j','h','ks'],[1.3,1.6,2.2],colors[0],markers[0],'2MASS')
	Spitzer = plots(['i1','i2','i3','i4','m1','m2'],[3.6,4.5,5.8,8.,24,70],colors[1],markers[1],'Spitzer')
	WISE = plots(['w1','w2','w3','w4'],[3.4,4.6,12,22],colors[2],markers[2],'WISE')
	SOFIA = plots(['F11','F19','F31','F37'],[11.1,19.7,31.5,37.1],colors[3],markers[3],'SOFIA')
	IRAS = plots(['Fnu_12','Fnu_25','Fnu_60','Fnu_100'],[12,25,60,100],colors[4],markers[4],'IRAS')
	AKARI = plots(['S65','S90','S140','S160'],[65,90,140,160],colors[5],markers[5],'AKARI')
	ENOCH = plots(['Fp'],[1300],colors[6],markers[6],'ENOCH')
	HERSCHEL = plots(['H70','H160','H250','H350','H500'],[70,160,250,350,500],colors[7],markers[7],'HERSCHEL')
	SCUBA = plots(['S450','S850','S1300'],[450,850,1300],colors[8],markers[8],'SCUBA')
	fluxlist = [TwoMASS,Spitzer,WISE,SOFIA,IRAS,AKARI,ENOCH,HERSCHEL,SCUBA]
	fluxnames = [p.bands for p in fluxlist]
	
	# loop on all the sources in sourcetable
	sources = sourcetable.group_by('SOFIA_name')
	for key,sourcetable in zip(sources.groups.keys,sources.groups):

		# enable the user to just plot the SEDs from one given cluster
		if cluster==None or sourcetable['Cluster'][0]==cluster:

			# pick the source name, RA and DEC
			sourceID = sourcetable['SOFIA_name'][0] 
			RA = sourcetable[RAstr][0]
			DEC = sourcetable[DECstr][0]
			
			# clean up existing png file (i don't think this step is necessary)
			os.system('rm %s' %(folder_export+sourceID+'.png'))

			# create a new 'master' figure
			fig = plt.figure()

			# add provision for the SED plot in the figure (see add_axes doc)
			ax = fig.add_axes([0.08,0.27,0.7,0.6])

			# create a numpy masked table with the fluxes
			ntable=nptable(sourcetable[[col for p in fluxlist for col in p.bands]][0])

			# counts the number of unmasked fluxes
			count=np.ma.count(nptable(sourcetable[[col for p in [TwoMASS,Spitzer,WISE,SOFIA] for col in p.bands]][0]))

			# this is the name of the file containing the fits results
			fname = folder_export+"plot_"+sourcetable['Cluster'][0]+'/'+sourceID+'.tab'

			# nonzero number of unmasked data points? worth it!
			if count:
				print "Generating plot for "+sourcetable['SOFIA_name'][0]
				# Plot the fit results and display some fits parameters
				data_list = []

				# only look for the fits for isolated and clustered sources were computed in the first place
				if (sourcetable['Property'][0] == 'Isolated' or sourcetable['Property'][0] == "Clustered") and count>2 and os.path.isfile(fname):
					SEDfolder = '/cardini3/mrizzo/2012SOFIA/SED_Models/'
					model_dir = SEDfolder+'models_r06/seds/'

					# plot the results of all fits for that source, that match the criterion 'select_format
					filename = SEDfolder+'output_'+sourcetable['Cluster'][0]+'.fitinfo'
					plotsedfit(filename,ax,select_format=('N',10.),sources=[sourceID])

					# load the fits result datafiles (created by sedfitter and related functions in sedfits.py)
					fitparams = pickle.load(open(fname,'r'))
					fitparams_min = pickle.load(open(folder_export+"plot_"+sourcetable['Cluster'][0]+'/'+sourceID+'_min.tab','r'))
					fitparams_max = pickle.load(open(folder_export+"plot_"+sourcetable['Cluster'][0]+'/'+sourceID+'_max.tab','r'))

					# This forms the list of all fit parameters and values from the fitter
 					data_list = [nametable[col][0]+": "+str(fitparams_min[col][0])+", "+str(fitparams[col][0])+", " \
						+str(fitparams_max[col][0]) for col in fitparams.columns[2:]] 

					# add composite values
					solar_lum = 3.846e26 # W		
					solar_radius = 6.955e8 # m
					solar_mass = 1.98e30 # kg
					year = 3.15569e7 # s
					Rstar = fitparams['rstar'][0]*solar_radius
					Tstar = fitparams['tstar'][0]
					Mdotdisk= fitparams['mdotdisk'][0] * solar_mass / year
					Mstar = fitparams['massc'][0] * solar_mass
					L_star = 4*np.pi*constants.sigma_sb.value*Rstar**2*Tstar**4/solar_lum
					L_acc_disk = constants.G.value*Mdotdisk*Mstar/Rstar/solar_lum
					data_list.append(r'$L_\star$ = %.2e'  %(L_star))
					data_list.append(r'$L_{disk}$ = %.2e' % (L_acc_disk))
					data_list.append(r'$L_{tot}$ = %.2e' % (L_acc_disk+L_star))

				# also add the R50 ratios to grasp whether some sources are extended or not
				data_list.append(r"$R_{50}^{19}$ = %.2f $R_{cal}^{19}$ " % (sourcetable['R50_19'][0]/sourcetable['R50_cal_19'][0]))
				data_list.append(r"$R_{50}^{31}$ = %.2f $R_{cal}^{31}$ " %(sourcetable['R50_31'][0]/sourcetable['R50_cal_31'][0]))
				data_list.append(r"$R_{50}^{37}$ = %.2f $R_{cal}^{37}$ " %(sourcetable['R50_37'][0]/sourcetable['R50_cal_37'][0]))
				
				# join the string
				data_string = "\n".join(data_list)

				# display all those parameters to the right of the main SED
				fig.text(0.78,0.95,data_string,ha='left',va='top',size=6)

				# plot all mission data in different color and marker; convert values to lambdaFlambda units
				for p in fluxlist:
					plotData(ax,sourcetable,p,alpha)

				# put the legend on the bottom right
				ax.legend(loc=4,fontsize='small')

				# log axes
				ax.set_xscale('log')
				ax.set_yscale('log')

				# axes limits
				ax.set_xlim([1,1400])
				ax.set_xticks([1,5,10,20,30,40,70,100,500,1300])

				# necessary to not have log notation
				ax.xaxis.set_major_formatter(ScalarFormatter())

				# labels
				ax.set_xlabel(r'Wavelength (microns)')
				ax.set_ylabel(r'$\lambda F_\lambda$ (ergs.cm$^{-2}$.s$^{-1}$)')

				# load up the 6 image cutouts underneath the main SED
				for j in range(7):
					# creates the axes for the cutout images					
					newax = fig.add_axes([0.02+j*0.13,0.04,0.14,0.14])

					# function newimgcutouts looks into all the FITS files for a match in RA and DEC
					# returns the sub image cutout around the source
					subimg,axval = newimgcutouts(newax,sourcetable,j+1)

					# function "label" determines the label of each cutout (e.g. "SOFIA 19um")
					label = newdet_label(j+1)	
		
					# set label fontsize
					newax.set_xlabel(label,fontsize=9)

					# no ticks
					newax.set_xticks([])
					newax.set_yticks([])

				# write the size of the cutout in arcsec
				straxval = "%.1f asec" % axval

				# add vertical text next to the last cutout
				newax.text(1,0.5,straxval,
					horizontalalignment = 'left',
					verticalalignment = 'center',
					rotation = -90,
					transform = newax.transAxes,fontsize=9)

				# show title
				ax.set_title(sourceID)
				
				# show and hold (if show==True), otherwise just save to png with high dpi
				if show: plt.show()
				
				fig.savefig(folder_export+sourceID+'.png',dpi=200)
				plt.close(fig)


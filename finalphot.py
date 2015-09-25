''' This is the master script that will reduce all of the SOFIA data '''

### IMPORTS AND DEFINITIONS ###
import photometry as p
from astropy.table import Table,vstack,join
import astropy.units as u
import os,re
import numpy as np
import pickle
import subprocess as sp
from itertools import izip
metafolder = "/cardini3/mrizzo/2012SOFIA/2014meta/"
folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"
folder_mosaics = "/n/a2/mrizzo/Dropbox/SOFIA/Mosaics/"
folder_data = "/cardini3/mrizzo/2012SOFIA/alldata/"
reduction_folder = "/cardini3/mrizzo/2012SOFIA/Reduction/"
fieldlist = ["CepA","CepC","IRAS20050","NGC2264","NGC1333","NGC7129","Oph","S140","S171","NGC2071"]

metalist = ["SOFIA_name",'Cluster','RA','DEC',"Property","SemiMajorAxis","SemiMinorAxis",\
				"R50_19","R50_31","R50_37","R50_cal_19","R50_cal_31","R50_cal_37"]
for cat in ['allwise','megeath','c2d','guth','iras','akari','akariyso','enoch06','enoch08','van_Kempen']:
	metalist.append('_RAJ2000_'+cat),metalist.append('_DEJ2000_'+cat)
columnlist = ['j','h','ks','w1','i1','i2','w2','i3','i4','F11','w3','Fnu_12','F19','w4','m1','Fnu_25','F31','F37',\
		'Fnu_60','S65','m2','H70','S90','Fnu_100','S140','S160','H160','H250','S450','H350','H500','S850','S1300','Fp']
errorbars = ["e_"+col for col in columnlist]
newwavelengths = [1.3,1.6,2.2,3.4,3.6,4.5,4.6,5.8,8.,11.1,12,12,19.7,22,24,25,31.5,37.1,60,65,70,71.9,90,100,140,160,167,250,350,450,500,850,1300,1300]

#####################################################################################################################################

print '### STEP 1: WORK ON CALIBRATORS ###'

# Make a table that categorizes all of the calibrator measurements taken over 10 flights
p.make_calibrator_table()

# For each calibrator, calculate the aperture correction associated with our wavelengths bands
p.make_calibrator_corr_table()

# Plot various diagnostics for the paper
p.plot_cal_profiles()

######################################################################################################################################

print  '### STEP 2: PROCESS THE RAW DATA ###'

# Call the MIRIAD routine that:
# 1: reads the images,
# 2: crops & adjust 
# 3: masks & calibration factors
# 4: subtracts background 
# 5: makes mosaics
os.system("/cardini3/mrizzo/anaconda/bin/python do_everything.py")

######################################################################################################################################

print '### STEP 3: IMPORT SOURCE TABLE ###'

source_fname = 'all_sources.reg'			# This is crafted manually in our case
table,table_bkgd = p.parse_new_source_file(metafolder+source_fname) 	# this also exports a .reg file to folder_export to be read out by DS9, with all of our sources and names

######################################################################################################################################

print '### STEP 4: BACKGROUND PHOTOMETRY SCRIPT ###'
photbkgd = p.photBkgd(table_bkgd)

# save
pickle.dump(photbkgd,open(folder_export+"photbkgd.data","wb"))
photbkgd.write(folder_export+'photbkgd.txt',format='ascii.fixed_width')

######################################################################################################################################

print '### STEP 5: MAIN PHOTOMETRY SCRIPT ###'
newPhot = p.newPhot(table)

# save after this step
os.system('rm %s %s' % (folder_export+'newPhot.txt', folder_export+"newPhot.data"))
pickle.dump(newPhot,open(folder_export+"newPhot.data","wb"))
newPhot.write(folder_export+'newPhot.txt',format='ascii.fixed_width')

######################################################################################################################################

print '### STEP 6: CROSS-REFERENCE WITH ONLINE TARGETS ###'
newPhot = pickle.load(open(folder_export+"newPhot.data","r"))
allNewPhot = p.search_vizier(newPhot)

# save
allNewPhot.write(folder_export+'allNewPhot.txt',format='ascii.fixed_width')
pickle.dump(allNewPhot,open(folder_export+"allNewPhot.data","wb"))

######################################################################################################################################

print '## STEP 7: CONVERTS RESULTING TABLE TO CORRECT UNITS & FORMATS ###'
allNewPhot = pickle.load(open(folder_export+"allNewPhot.data","r"))
newtable = p.convert_table(allNewPhot)

# save
newtable.write(folder_export+'newtable.txt',format='ascii.fixed_width')
pickle.dump(newtable,open(folder_export+"newtable.data","wb"))

######################################################################################################################################

print '### STEP 8: ADD HERSCHEL PHOTOMETRY ###'
newtable = pickle.load(open(folder_export+"newtable.data","r"))
newsourcelist = p.addHerschel(newtable)

# save
pickle.dump(newsourcelist,open(folder_export+"newsourcelist.data","wb"))

# Write a file with only the fluxes
newsourcelist[metalist+columnlist+errorbars].write(folder_export+"newTableReduced.txt",format='ascii.fixed_width')

#####################################################################################################################################

print '### STEP 8.5: DO IRAC20050 PHOTOMETRY ###'
# scripts outputs newtable.data
os.system("/cardini3/mrizzo/anaconda/bin/python IRAS20050.py")

######################################################################################################################################

print '### STEP 9: CALCULATE SPECTRAL INDICES ###'
totsourcetable = pickle.load(open(folder_export+"newtable.data","r"))
newsourcelist = totsourcetable.copy()

# adjusts error bars determination to determine the spectral index (otherwise the Spitzer fluxes constrain the fits too much)
#for col in columnlist:
#	if "F" not in col and if newsourcelist["e_"+col]<0.1*newsourcelist[col]:
#		newsourcelist["e_"+col] = 0.1*newsourcelist[col] # imposes 10% minmum uncertainty everywhere

# spectral index is only determined from ~2-20um
totsourcetable_wrong_errors = p.calcAlpha(newsourcelist[metalist+columnlist+errorbars],columnlist[2:-19],errorbars[2:-19],newwavelengths[2:-19])
totsourcetable['alpha'] = totsourcetable_wrong_errors['alpha']

# save
pickle.dump(totsourcetable,open(folder_export+"totsourcetable.data","wb"))
totsourcetable.write(folder_export+"totsourcetable.txt",format='ascii.fixed_width')

#####################################################################################################################################

print '### STEP 10: FIT ALL SEDs USING SEDFITTER ###'
totsourcetable=pickle.load(open(folder_export+"totsourcetable.data","r"))

# import package
import sedfits

# convolve models with filters (only need to do this once)
#sedfits.convolve_models()

# fit
sedfits.fit_all_sources(totsourcetable)

# add some of the fit results to the master table
totsourcetable_fits = p.addFitResults(totsourcetable,folder_export=folder_export)

# save
pickle.dump(totsourcetable_fits,open(folder_export+"totsourcetable_fits.data","wb"))
totsourcetable_fits.write(folder_export+"totsourcetable_fits.txt",format='ascii.fixed_width')

#####################################################################################################################################

print '### STEP 11: plot all SEDs ###'
totsourcetable_fits = pickle.load(open(folder_export+"totsourcetable_fits.data","r"))

# sort the sources by property (Isolated, Extended, or Clustered)
sourcetype=totsourcetable_fits.group_by("Property")

# for each source in each group
for key,group in izip(sourcetype.groups.keys,sourcetype.groups):
	
	# export a condensed table with only the sources of this group (e.g. all isolated sources)
	pickle.dump(group[metalist+columnlist+errorbars],open(folder_export+"table_"+key["Property"]+".data","wb"))
	group[metalist+columnlist+errorbars].write(folder_export+"table_"+key["Property"]+".txt",format="ascii.fixed_width")
	
	# create PNGs of the fits with all the extra flux measurements from the archives
	p.markerPlotSED(group[metalist+columnlist+errorbars],show=False,folder_export=folder_export,RAstr="RA",DECstr="DEC")


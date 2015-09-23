''' this script is designed to fit the SEDs obtained from newphot.py
using Tom Robitaille SED fitter tool '''
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import sedfitter
import astropy.units as u
import astropy.constants as const
import pickle
from astropy.table import Table,vstack

from itertools import izip
from sedfitter import plot,plot_params_1d,plot_params_2d
from sedfitter import fit,write_parameters,write_parameter_ranges,filter_output
import os


folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"

SEDfolder = '/cardini3/mrizzo/2012SOFIA/SED_Models/'
model_dir = SEDfolder+'models_r06'

def convolve_models():
	'''
	This function convolves Robitaille's models with the instrument bandpasses that we have
	DONT FORGET TO REMOVE EXISTING MODEL FILES BEFORE RUNNING SCRIPT
	'''
	# STEP 1: import the SOFIA band profiles from Tracy's files

#	for filename in ['11pt3mu.txt']:#,'Herschel_Pacs.blue.dat','Herschel_Pacs.red.dat','Herschel_SPIRE.PSW.dat','Herschel_SPIRE.PMW.dat','Herschel_SPIRE.PLW.dat']: \
#		hersch = np.loadtxt(filename); print hersch.shape; np.savetxt(filename,np.array([hersch[::-1,0],hersch[::-1,1]]).T)

	from sedfitter.filter import Filter

	# the following convolves the existing SEDs with the normalized response of each filter
	# this should only need to be done once, as it creates files in the models folder that can be used later

	filt11 = Filter()
	filt11= Filter.read(SEDfolder+'11pt3mu.txt')
	filt11.name = "F11"
	filt11.normalize()

	PACS70 = Filter()
	PACS70= Filter.read(SEDfolder+'Herschel_Pacs.blue.dat')
	PACS70.name = "H70"
	PACS70.normalize()

	PACS160 = Filter()
	PACS160= Filter.read(SEDfolder+'Herschel_Pacs.red.dat')
	PACS160.name = "H160"
	PACS160.normalize()

	SPIRE250 = Filter()
	SPIRE250= Filter.read(SEDfolder+'Herschel_SPIRE.PSW.dat')
	SPIRE250.name = "H250"
	SPIRE250.normalize()

	SPIRE350 = Filter()
	SPIRE350= Filter.read(SEDfolder+'Herschel_SPIRE.PMW.dat')
	SPIRE350.name = "H350"
	SPIRE350.normalize()

	SPIRE500 = Filter()
	SPIRE500= Filter.read(SEDfolder+'Herschel_SPIRE.PLW.dat')
	SPIRE500.name = "H500"
	SPIRE500.normalize()

	filt19 = Filter()
	filt19= Filter.read(SEDfolder+'FORCAST_19.7um_dichroic.txt')
	filt19.name = "F19"
	filt19.normalize()

	filt31 = Filter()
	filt31=Filter.read(SEDfolder+'FORCAST_31.5um_dichroic.txt')
	filt31.name = "F31"
	filt31.normalize()

	filt37 = Filter()
	filt37=Filter.read(SEDfolder+'FORCAST_37.1um_dichroic.txt')
	filt37.name = "F37"
	filt37.normalize()
	
	from sedfitter.convolve import convolve_model_dir

	convolve_model_dir(model_dir,[filt11])

def fit_all_sources(sourcetable):
	'''
	Wrap around Robitaille's fitting routine
	'''

	from sedfitter.extinction import Extinction
	# load up extinction law as suggested by website; might switch over to other dust files
	extinction = Extinction.from_file(SEDfolder+'extinction_law.ascii',columns=[0,1],wav_unit=u.micron,chi_unit=u.cm**2/u.g)

	# only fit the isolated and clustered sources
	types = sourcetable.group_by('Property')
	totsourcetable = vstack([types.groups[0],types.groups[2]])

	# list of columns with fluxes to be used to fit
	columnlist = ['j','h','ks','i1','i2','i3','i4','F11','F19','m1','F31','F37','m2','S450','S850','H70','H160','H250','H350','H500']
	errorlist = ["e_"+col for col in columnlist]
	flaglist = ["flag_"+col for col in columnlist if "flag_"+col in totsourcetable.columns]

	# set up input table to be fed to sedfitter
#	fluxlist = []
#	for col in columnlist:
#		fluxlist.append(col);fluxlist.append('e_'+col)

	# cluster names and distances
	fieldlist = Table(names=["CepA","CepC","IRAS20050","NGC2264","NGC1333","NGC7129","Oph","S140","S171","NGC2071"])
	fieldlist.add_row([730,730,700,760,240,1000,160,900,850,490])

	# for each cluster
	clusters = totsourcetable.group_by('Cluster')

	for key,group in izip(clusters.groups.keys,clusters.groups):

		# isolate source and use only relevant columns
		newsourcetable = group[['SOFIA_name','RA','DEC']+columnlist+errorlist+flaglist]
		print "Working on cluster ",key['Cluster']
		print newsourcetable

		# number of sources in that cluster
		L = len(newsourcetable)
		
		# for each column
		for col in columnlist:
			# re-check that there are no negative fluxes that are not masked
			for i in range(len(newsourcetable)):
				if newsourcetable[col][i]<0:
					newsourcetable.mask[col][i] = True

			# create the flag table based on mask
			newsourcetable["flag"+col] = np.ma.array(~newsourcetable[col].mask).astype(int)

			# make sure the errors are flagged as well
			newsourcetable['e_'+col].mask = newsourcetable[col].mask


		# multiplies the flag by 3 for the Herschel fluxes that we know are taken with a very large aperture
		for col in ['H250','H350','H500']:
			newsourcetable["flag"+col] *= 3

		for col in columnlist:
			for i in range(len(newsourcetable)):
				# if flag string contains a 'U', then the flux is an upper limit
				if 'flag_'+col in newsourcetable.columns:
					if 'U' in newsourcetable['flag_'+col][i] and newsourcetable["flag"+col][i] != 3:
						newsourcetable["flag"+col][i] *= 3

		# convert to mJy
		for col in columnlist:
			newsourcetable[col] *= 1000.0
			newsourcetable['e_'+col] *= 1000.0
		
		# creates the list of columns containing the flags
		newflaglist = ["flag"+col for col in columnlist]

		# create proper flux;error lists
		fluxlist = []
		for col in columnlist:
			fluxlist.append(col);fluxlist.append('e_'+col)
		

		# creates the table in proper format for feeding to sedfitter
		final = newsourcetable[['SOFIA_name','RA','DEC']+newflaglist+fluxlist].filled(-1.0)
		final.write(folder_export+key['Cluster']+'_sedfitter.tab',format='ascii')

		# remove first line (column headers)
		os.system('sed -i -e "1d" %s' % (folder_export+key['Cluster']+'_sedfitter.tab'))

		# name of the filter names for use in sedfitter
		sednamelist = ['2J','2H','2K','I1','I2','I3','I4','F11','F19','M1','F31','F37','M2','W1','W2','H70','H160','H250','H350','H500']

		# list of aperture sizes
		apertures = [8,8,8,12,12,12,12,9,9,35,9,9,45,20,40,12,22,22,30,42] *u.arcsec ### CHECK APERTURE SIZES  ,12,22,22,30,42

		# distance to the current cluster
		distance = fieldlist[key['Cluster']][0]
		
		# cleans up the directories before starting
		os.system('rm -rf %s' % (folder_export+'plot_'+key['Cluster']))
		os.system('rm %s' % (SEDfolder+'output_'+key['Cluster']+'.fitinfo'))

		# fitting routine
		fit(folder_export+key['Cluster']+'_sedfitter.tab',sednamelist,apertures,
			model_dir,SEDfolder+'output_'+key['Cluster']+'.fitinfo',
			extinction_law = extinction,
			distance_range = [distance*0.8,distance*1.2] * u.pc,
			av_range = [0.,10])

		print "Generating some plots"
		plot(SEDfolder+'output_'+key['Cluster']+'.fitinfo',folder_export+'plot_'+key['Cluster'],select_format=('N',10.))
		#plot_params_1d(SEDfolder+'output_'+key['Cluster']+'.fitinfo','MDISK',output_dir=folder_export+'plot_'+key['Cluster']+"/1d",log_x=True,select_format=('F',3.))
		print "Extracting the parameters from the fits"
		write_parameters(SEDfolder+'output_'+key['Cluster']+'.fitinfo',folder_export+'plot_'+key['Cluster']+'/params.txt',select_format=('N',10.))
		write_parameter_ranges(SEDfolder+'output_'+key['Cluster']+'.fitinfo',folder_export+'plot_'+key['Cluster']+'/params_ranges.txt',select_format=('N',10.))
		print "Parsing results..."
		parse_params_table(folder_export+'plot_'+key['Cluster'])
		print "Done parsing results"
		
def parse_params_table(fileloc):
	'''
	This parses one of the parameter files that is given out by Tom Robitaille's SED fitter
	1) Open up the file
	2) discard the first few lines (headers)
	3) read the first source line, and record the number of fits N
	4) read the next N lines and populate an astropy table (one per source)
	5) for each source, write down the list of fit numbers, the Av and the scale and save them to a separate astropy table used for overplotting onto our SEDs
	'''
	col_names = ['fit_id', 'model_name', 'chi2', 'av', 'scale', 'time', 'massc', 'rstar', 'tstar',\
			 'mdot', 'rmax', 'theta', 'rmine', 'mdisk', 'rmaxd', 'rmind', 'rmind(au)', 'rc',\
			 'rchole', 'zmin', 'a', 'b', 'alpha', 'rhoconst', 'rhoamb', 'mdotdisk', 'incl.',\
			 'av_int', 'tau60', 'ltot', 'h100']
	col_names_range = ['SOFIA_name', 'n_data', 'n_fits', 'chi2', 'av', 'scale', 'time', 'massc', 'rstar',\
			 'tstar', 'mdot', 'rmax', 'theta', 'rmine', 'mdisk', 'rmaxd', 'rmind', 'rmind(au)',\
			 'rc', 'rchole', 'zmin', 'a', 'b', 'alpha', 'rhoconst', 'rhoamb', 'mdotdisk', 'incl.',\
			 'av_int', 'tau60', 'ltot', 'h100']

	# set types for tables
	dtype = ['<f8' for col in col_names]
	dtype_range = ['<f8' for col in col_names_range]
	dtype[1] = '<S12'
	dtype_range[0] = '<S12'

	# inits
	paramsfile = fileloc+'/params.txt'
	paramsrangefile = fileloc+'/params_ranges.txt'

	# open files while skip first rows
	f = open(paramsfile)
	lines = f.readlines()[3:]
	f.close()
	f = open(paramsrangefile)
	rangelines = f.readlines()[3:]
	f.close()

	# read all the lines for the fit parameter results
	i = 0
	while i<len(lines):
		
		sline = lines[i].split()

		# this is the start of a new fit block
		if len(sline)==3:

			# grab nfit value, name, cluster
			nfit = int(sline[-1])
			name = sline[0]
			cluster = sline[0].split('.')[0]

			# create table
			table = Table(names = col_names,dtype=dtype)

			# read all fit content
			for j in range(1,nfit+1):
				newline = lines[i+j].split()
				table.add_row(newline)

			# export
			pickle.dump(table,open(folder_export+"plot_"+cluster+'/'+name+'.tab','wb'))
			table.write(folder_export+"plot_"+cluster+'/'+name+'.txt',format='ascii')

			# jump to next line
			i +=nfit
		else: 
			i+=1

	# read all the lines for the fir parameter range results
	i=0
	while i<len(rangelines):
		sline = rangelines[i].split()
		name = sline[0]
		cluster = sline[0].split('.')[0]

		# create a table for the minimum values
		table_min = Table(names=col_names_range,dtype=dtype_range)

		# record the first minimum in table
		newrow = [sline[0],int(sline[1]),int(sline[2])]

		# record all the rest of the minimums
		newrow += [sline[ncol] for ncol in np.arange(3,3*(len(col_names_range)-3)+1,3)]

		# add it to the table
		table_min.add_row(newrow)

		# save
		pickle.dump(table_min,open(folder_export+"plot_"+cluster+'/'+name+'_min.tab','wb'))

		# repeat for maximums
		table_max = Table(names=col_names_range,dtype=dtype_range)
		newrow = [sline[0],int(sline[1]),int(sline[2])]
		newrow += [sline[ncol] for ncol in np.arange(5,3*(len(col_names_range)-3)+5,3)]
		table_max.add_row(newrow)
		pickle.dump(table_max,open(folder_export+"plot_"+cluster+'/'+name+'_max.tab','wb'))
		i+=1



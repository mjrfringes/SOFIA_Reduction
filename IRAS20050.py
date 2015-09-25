'''
This file does the processing of the IRAS20050 cluster observations.
'''
import photometry as p
import pickle


metafolder = "/cardini3/mrizzo/2012SOFIA/2014meta/"
folder_export = "/n/a2/mrizzo/Dropbox/SOFIA/Processed_Data/"
folder_mosaics = "/n/a2/mrizzo/Dropbox/SOFIA/Mosaics/"
folder_data = "/cardini3/mrizzo/2012SOFIA/alldata/"
reduction_folder = "/cardini3/mrizzo/2012SOFIA/Reduction/"
fieldlist = ["CepA","CepC","IRAS20050","NGC2264","NGC1333","NGC7129","Oph","S140","S171","NGC2071"]



######################################################################################################################################

### STEP 1: LOAD PHOTOMETRY TABLE FROM MASTER SCRIPT ###
totsourcetable = pickle.load(open(folder_export+"newsourcelist.data","r"))

######################################################################################################################################

### STEP 2: REPLACE IRAC PHOTOMETRY FOR IRAS20050 ###
newIRACphot = p.do_IRAC_phot(totsourcetable,'IRAS20050')

# column selection
metalist = ["SOFIA_name",'Cluster','RA','DEC',"Property"]\
		#"SemiMajorAxis","SemiMinorAxis","R50_19","R50_31","R50_37","R50_cal_19","R50_cal_31","R50_cal_37"]
columnlist=['j','h','ks','i1','i2','i3','i4']#,'F11','F19','F31','F37']
errorlist = ["e_"+col for col in columnlist]
flaglist = ["flag_"+col for col in columnlist if "flag_"+col in newIRACphot.columns]

# Adjust formatting of table for printing
for col in columnlist+errorlist:
	newIRACphot[col].format = '%.3f'

######################################################################################################################################

### STEP 3: MANUAL ADJUSTMENT OF THE 2MASS PHOTOMETRY ###
# this occurs after looking at the 2MASS images and setting upper limits
newtable = p.do_manual_2mass_phot(newIRACphot)

# print
#newtable[metalist+columnlist+errorlist+flaglist].more()

# save
pickle.dump(newtable,open(folder_export+"newtable.data","wb"))


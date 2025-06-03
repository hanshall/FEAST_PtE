"""
    production_emission_data loads emissions data from numerous component level studies that are compiled in the
    ProductionSite-ComponentEmissions.xlsx spreadsheet.
"""

# -------- Modifying the pickling and formatting functionality for the emissions distribution file 
#   This was an example code from the FEAST files, that I'm modifying to make a one-off. 
# from: production_emission_data.py (FEAST_PtE/ExampleData/RawDataProcessingScripts/)

#Note: because I am doing this the dumb way, I had to move this script to the project parent folder (FEAST_PtE); I just moved it back to this folder for simplicity. 

# -------------- reading the spreadsheet --------------
import pandas as pd
from feast.input_data_classes import LeakData
import pickle
import numpy as np
from os.path import dirname, abspath
import os


# ----------- Set up and load in the files I want to modify and pickle 

file_in_leaks = 'EPA_Trial_Run/Setup_Files/Comp_Leaks.csv'
file_in_tanks = 'EPA_Trial_Run/Setup_Files/Comp_Leaks.csv'
file_in_super = 'EPA_Trial_Run/Setup_Files/Comp_Leaks.csv'

file_out_leaks = 'EPA_Trial_Run/Setup_Files/Comp_Leaks.p'
file_out_tanks = 'EPA_Trial_Run/Setup_Files/Comp_Tanks.p'
file_out_super = 'EPA_Trial_Run/Setup_Files/Comp_Super.p'

#yes I know that this is the dumbest way to do this, it's fine. I'm babystepping through python on this one. 

comp_leaks = pd.read_csv(file_in_leaks)
comp_tanks = pd.read_csv(file_in_tanks)
comp_super = pd.read_csv(file_in_super)

#------------ Modify to Data Arrays, remove the NA values,and flatten distribution files

comp_leaks = np.array(comp_leaks)
comp_tanks = np.array(comp_tanks)
comp_super = np.array(comp_super)

comp_leaks = comp_leaks[np.invert(np.isnan(comp_leaks))]
comp_tanks = comp_tanks[np.invert(np.isnan(comp_tanks))]
comp_super = comp_super[np.invert(np.isnan(comp_super))]
#Okay, that sound give us a flatted list with no NANs; 
#FRom what Ican tell from the Docket, this data is all already in g/s, so no need to adjust, but the orignal conversion files included:
# em_array = em_array * 1000 / 24 / 3600  # convert from kg/day to g/s

#Do due diligencence and remove any zeros. 
comp_leaks = comp_leaks[comp_leaks > 0]
comp_tanks = comp_tanks[comp_tanks > 0]
comp_super = comp_super[comp_super > 0]

#alright, cool. 
#As noted this was the stupid and hard coded way to do this, but whatever - someone else can fix it later if desired. 

#----------run the conversion portions of the code. I pulled this directly from the orginal FEAST Example, I don't know if this assumption of 650 comp/well holds for the RTI distribvutions


notes = \
    """
    Data extracted from the compilation spreadsheet ProductionSite-ComponentEmissions.xlsx
    The number of components surveyed at each well generally were not recorded. Therefore, the number of components is 
    estimated by assuming 650 components per well.    
    """


#----Leaks: 

emissions_leaks = LeakData(notes=notes, raw_file_name=file_in_leaks.split('/')[-1], data_prep_file='emission_distribution_formatting.py')

# The dict structure allows for multiple types of detection methods used in the study
leak_data = {'All': comp_leaks}
well_counts = {'All': 2612}  # Number of wells in the study
comp_counts = {'All': 650}  # Assumed components per well

emissions_leaks.define_data(leak_data=leak_data, well_counts=well_counts, comp_counts=comp_counts)

pickle.dump(emissions_leaks, open(file_out_leaks, 'wb'))


#----Tanks: 

emissions_tanks = LeakData(notes=notes, raw_file_name=file_in_tanks.split('/')[-1], data_prep_file='emission_distribution_formatting.py')

# The dict structure allows for multiple types of detection methods used in the study
leak_data = {'All': comp_tanks}
well_counts = {'All': 2612}  # Number of wells in the study
comp_counts = {'All': 650}  # Assumed components per well

emissions_tanks.define_data(leak_data=leak_data, well_counts=well_counts, comp_counts=comp_counts)

pickle.dump(emissions_tanks, open(file_out_tanks, 'wb'))

#----Tanks: 

emissions_super = LeakData(notes=notes, raw_file_name=file_in_super.split('/')[-1], data_prep_file='emission_distribution_formatting.py')

# The dict structure allows for multiple types of detection methods used in the study
leak_data = {'All': comp_super}
well_counts = {'All': 2612}  # Number of wells in the study
comp_counts = {'All': 650}  # Assumed components per well

emissions_super.define_data(leak_data=leak_data, well_counts=well_counts, comp_counts=comp_counts)

pickle.dump(emissions_super, open(file_out_super, 'wb'))


print('Successfully completed production-emission-data processing.')

"""
The Basic EPA Re-consideration Feast Modeling Script. 
This builds from the example script from RTI and the FEAST example run file. 

Running: Python 3.12.4 

V1.0 Date: 2025-06
v1.5 Date: 2025-06
v1.6 Date: 2025-06

Updates Tracking List: 
V1.5 - adding copy.deepcopy to all of the called variables to prevent crosstalk, especially in the LDAR definitions. 
V1.6 - updating parameters to reflect setup from UT (Arvind, Haojun) Including modeing out super emitters as episodic emissions 
V1.7 - Regressed the super emitters as periodic emissions build - that sets the super emitters as non-fixable. Also updated the distributions to the correct values. 
"""
import numpy as np #v2.2.3
import copy
import feast.EmissionSimModules.infrastructure_classes
import feast.EmissionSimModules.simulation_classes as sc
from feast.EmissionSimModules.infrastructure_classes import Component
import feast.DetectionModules as Dm
import pickle
import time
import json
import os
from scipy.special import erf #v1.15.2

###############################################################################
# USER INPUT
sim_years = 5 #how many years should the simulation be run for? - Note- delta is 1 day, set inside define_time_settings  
n_montecarlo = 10 #How many iterations of the model are we running? 
write_out_location = 'EPA_Trial_Run/check'#'EPA_Trial_Run/run_results_loop_test_corrected_distributions'
Set_periodic_threshold = 5 #detection threshold of the periodic survey in kg/hr 

#LOOPING INSTRUCTIONS
cadence_values = {'monthly': 30, 'bimonthly': 60, 'quarterly': 91, 'semiannual':182, 'annual': 365}

#GAS FIELD SETUP 
mpStr = 'Model Plant 4' #Which model plant are we using in the simulation? 
nSites = 20 #how many model plants in the Gas Field - note - double check placement of varible in code.
met_data_loc = 'ExampleData/TMY-DataExample.csv'

# EMITTERS SETUP 
repair_cost_distribution = 'EPA_Trial_Run/Setup_Files/fernandez_leak_repair_costs_2006.p'
#All Emitters are set to base_repairable, with a null repair rate = 0 

# General Leaks 
emission_distribution_leaks = 'EPA_Trial_Run/Setup_Files/Comp_Leaks.p'
comp_emissions_start_leaks = ((0.5/96)+ 0.0000001) #Fraction of components expected to be emissting at the beginnging of the simulation
comp_emissions_rate_leaks = ((0.5/100) / 365) # number of new emissions per component per day

#Tank Leaks (midsized low pressure equiptment)
emission_distribution_tanks = 'EPA_Trial_Run/Setup_Files/Comp_Tanks.p'
comp_emissions_start_tanks = ((3/96) + 0.0000001) #Fraction of components expected to be emissting at the beginnging of the simulation
comp_emissions_rate_tanks = ((2.5/100) / 365) # number of new emissions per component per day

#Super Emitters 
emission_distribution_super = 'EPA_Trial_Run/Setup_Files/Comp_Super.p' #A list of emission sizes to draw from for episodic emissions (g/s)
# emission_distribution_super = json.load(open('EPA_Trial_Run/Setup_Files/Comp_Super.json' ))#A list of emission sizes to draw from for episodic emissions (g/s)
# comp_emissions_per_day = ((5/100) / 365 ) #The average frequency at which episodic emissions occur (1/days)
# comp_emissions_duration = 0.5  #The duration of episodic emissions (days)
comp_emissions_start_super = ((0.5/96)+ 0.0000001) #Fraction of components expected to be emissting at the beginnging of the simulation
comp_emissions_rate_super = ((1.5/100) / 365) # number of new emissions per component per day

#TECHNOLOGY SETUP 
# All technologies have: detection_variables={'flux': 'mean'}, and site_queue = [] 
#Repair Delays
OGI_rep_delay = 30 #OGI repair delay in days
survey_rep_delay = 35 #Periodic survey repair delay in days 
sat_rep_delay = 45 #Satellite survey repair delay in days 

#OGI Settings - General
survey_speed_comp_ogi = 500 #Survey speed in components/hour 
suvey_labor_cost_ogi = 100 #survey cost USD/HR 
survey_ophrs_ogi = {'begin': 8, 'end': 17}
ogi_detection_points= np.array([0.0042, 0.0063, 0.0083, 0.012, 0.0166]) #Detection distribution, g/s
ogi_detection_probabilities= np.array([0, 0.25, 0.5, 0.75, 1]) #Detecion Distribution, probabilities 

#Periodic Survey Settings - General 
# survey_interval_periodic = 30 #return period for the tech in days
survey_sites_periodic = 100 #how many sites are observed each survey day 
survey_cost_periodic = 100 #USD/Site cost for the survey
survey_ophrs_periodic = {'begin': 8, 'end': 17}
periodic_detection_points = np.array([0.069, 0.105, 0.139, 0.208, 0.278]) * Set_periodic_threshold #detection Distribution, g/s
periodic_detection_probabilities = np.array([0, 0.25, 0.5, 0.75, 1]) #Detecion Distribution, probabilities

#Periodic Survey Settings - Large Event (satellite proxy) 
survey_interval_sat = 120 #return period for the tech in days
survey_sites_sat = 500 #how many sites are observed each survey day 
survey_cost_sat = 100 #USD/Site cost for the survey
survey_ophrs_sat = {'begin': 9, 'end': 16}
sat_detection_points = [6.94, 10.4, 13.9, 20.8, 27.8] #detection Distribution, g/s (set 100% at 100 kg/hr)
sat_detection_probabilities = np.array([0, 0.25, 0.5, 0.75, 1]) #Detecion Distribution, probabilities



###############################################################################



a = time.time()

def define_emitters():
    """
    Defines all emitters to be used in the simulation using the Component class
    Defining three classes, with underlying emissions distributions: General emitters (leaks), midsized low pressure emissions (tanks), and super emitters
    """
    # Generates reparable fugitive emissions
    comp_leak = feast.EmissionSimModules.infrastructure_classes.Component(
        name='Comp_Leaks',
        emission_data_path= copy.deepcopy(emission_distribution_leaks),
        emission_per_comp= copy.deepcopy(comp_emissions_start_leaks),  # Fraction of components expected to be emitting at the beginning of the simulation.
        emission_production_rate= copy.deepcopy(comp_emissions_rate_leaks),  # number of new emissions per component per day
        repair_cost_path= copy.deepcopy(repair_cost_distribution),
        base_reparable=True,
        null_repair_rate = 0.00000
    )
    
    # Generates reparable tank emissions
    comp_tank = feast.EmissionSimModules.infrastructure_classes.Component(
        name='Comp_Tanks',
        emission_data_path= copy.deepcopy(emission_distribution_tanks),
        emission_per_comp= copy.deepcopy(comp_emissions_start_tanks),  # Fraction of components expected to be emitting at the beginning of the simulation.
        emission_production_rate= copy.deepcopy(comp_emissions_rate_tanks),  # number of new emissions per component per day
        repair_cost_path= copy.deepcopy(repair_cost_distribution),
        base_reparable= True,
        null_repair_rate = 0.00000
    )

    # Generates reparable Super Emitter emissions
    comp_super = feast.EmissionSimModules.infrastructure_classes.Component(
        name='Comp_Super',
        emission_data_path= emission_distribution_super,
        # episodic_emission_sizes= copy.deepcopy(emission_distribution_super), # no inputs to the emission_data_path, instead point distribution to episodi emissions.
        # episodic_emission_per_day = copy.deepcopy(comp_emissions_per_day), #average frequency of the episodic emissions/day
        # episodic_emission_duration = copy.deepcopy(comp_emissions_duration), #how long are the periodic super emitters lasting.
        emission_per_comp= copy.deepcopy(comp_emissions_start_super),  # Fraction of components expected to be emitting at the beginning of the simulation.
        emission_production_rate= copy.deepcopy(comp_emissions_rate_super),  # number of new emissions per component per day
        repair_cost_path= copy.deepcopy(repair_cost_distribution),
        base_reparable= True,
        null_repair_rate = 0.00000
    )

    return comp_leak, comp_tank, comp_super


def define_sites(comp_leak, comp_tank, comp_super, modelPlantString):
    """
    Defines all sites to be used in the simulation. Sites consist of a collection of Component objects

    """
    
    # nSites = 20
    site_dict = {}
    
    if modelPlantString == 'Model Plant 1':
        newSite = feast.EmissionSimModules.infrastructure_classes.Site(
            name = modelPlantString,
            comp_dict = {'Leaks': {'number': 112, 'parameters': copy.copy(comp_leak)},
                         'Tanks': {'number': 0, 'parameters': copy.copy(comp_tank)},
                         'Super': {'number': 1, 'parameters': copy.copy(comp_super)}
                })
    elif modelPlantString == 'Model Plant 2':
        newSite = feast.EmissionSimModules.infrastructure_classes.Site(
                name = modelPlantString,
                comp_dict = {'Leaks': {'number': 220, 'parameters': copy.copy(comp_leak)},
                             'Tanks': {'number': 0, 'parameters': copy.copy(comp_tank)},
                             'Super': {'number': 1, 'parameters': copy.copy(comp_super)}
                    })
    elif modelPlantString == 'Model Plant 3':
        newSite = feast.EmissionSimModules.infrastructure_classes.Site(
                name = modelPlantString,
                comp_dict = {'Leaks': {'number': 612, 'parameters': copy.copy(comp_leak)},
                             'Tanks': {'number': 0, 'parameters': copy.copy(comp_tank)},
                             'Super': {'number': 2, 'parameters': copy.copy(comp_super)}
                    })
    elif modelPlantString == 'Model Plant 4':
        newSite = feast.EmissionSimModules.infrastructure_classes.Site(
                name = modelPlantString,
                comp_dict = {'Leaks': {'number': 612, 'parameters': copy.copy(comp_leak)},
                             'Tanks': {'number': 4, 'parameters': copy.copy(comp_tank)},
                             'Super': {'number': 2, 'parameters': copy.copy(comp_super)}
                    })
            
    site_dict[modelPlantString] = {'number': nSites, 'parameters': newSite} #what should be 'number' here?        

    return site_dict


def define_time_settings():
    """
    Sets the time resolution and duration of the simulation
    :return: a Time object containing simulation settings
    """
    return feast.EmissionSimModules.simulation_classes.Time(delta_t=1, end_time=365*sim_years)


def define_gas_field(timeobj, site_dict):
    """
    Creates a GasField object to be used in the simulation
    :param timeobj: A time object containing time settings for the simulation
    :param site_dict: A dict of sites to be included in the gas field
    :return gas_field: A GasField object to be used in the simulation
    """
    gas_field = feast.EmissionSimModules.infrastructure_classes.GasField(
        sites=site_dict,
        time=timeobj
    )
    gas_field.met_data_path = met_data_loc 
    gas_field.met_data_maker()
    return gas_field


def define_detection_methods(timeobj):
    """
    Define detection methods to be used in LDAR programs
    :param timeobj: A time object for simulation settings
    :return RD_ogi: repair delay for OGI, set days between detection and repair
    :return RD_survey: repair delay for OGI, set days between detection and repair
    :return RD_satellite: repair delay for OGI, set days between detection and repair 
    :return ogi_quarterly: a component survey representing quarterly ogi surveys.
    :return ogi_called_survey: a component survey representing ogi called by a periodic technology
    :return ogi_called_survey_sat: a component survey representing ogi called by a satellite overflight
    :return periodic_survey: a site survey representing a periodic technology
    :return large_event_survey: a site survey representing a satellite flyover 

    """
    #Setting the return delays: 
    RD_ogi = Dm.repair.Repair(repair_delay = copy.deepcopy(OGI_rep_delay))
    RD_survey = Dm.repair.Repair(repair_delay = copy.deepcopy(survey_rep_delay))
    RD_satellite = Dm.repair.Repair(repair_delay = copy.deepcopy(sat_rep_delay))


    
    ogi_quarterly = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=91,
        survey_speed= copy.deepcopy(survey_speed_comp_ogi),
        ophrs= copy.deepcopy(survey_ophrs_ogi),
        labor= copy.deepcopy(suvey_labor_cost_ogi),
        detection_variables={'flux': 'mean'},
        detection_probability_points= copy.deepcopy(ogi_detection_points),
        detection_probabilities= copy.deepcopy(ogi_detection_probabilities),
        dispatch_object= copy.deepcopy(RD_ogi),
        site_queue=[],
    )

    ogi_called_survey = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=None,
        survey_speed=copy.deepcopy(survey_speed_comp_ogi),
        ophrs=copy.deepcopy(survey_ophrs_ogi),
        labor=copy.deepcopy(suvey_labor_cost_ogi),
        detection_variables={'flux': 'mean'},
        detection_probability_points=copy.deepcopy(ogi_detection_points),
        detection_probabilities= copy.deepcopy(ogi_detection_probabilities),
        dispatch_object= copy.deepcopy(RD_survey),
        site_queue=[],
    )

    ogi_called_survey_sat = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=None,
        survey_speed=copy.deepcopy(survey_speed_comp_ogi),
        ophrs=copy.deepcopy(survey_ophrs_ogi),
        labor=copy.deepcopy(suvey_labor_cost_ogi),
        detection_variables={'flux': 'mean'},
        detection_probability_points=copy.deepcopy(ogi_detection_points),
        detection_probabilities= copy.deepcopy(ogi_detection_probabilities),
        dispatch_object= copy.deepcopy(RD_satellite),
        site_queue=[],
    )
    
    ogi_annual_survey = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=365,
        survey_speed=copy.deepcopy(survey_speed_comp_ogi),
        ophrs=copy.deepcopy(survey_ophrs_ogi),
        labor=copy.deepcopy(suvey_labor_cost_ogi),
        detection_variables={'flux': 'mean'},
        detection_probability_points=copy.deepcopy(ogi_detection_points),
        detection_probabilities= copy.deepcopy(ogi_detection_probabilities),
        dispatch_object= copy.deepcopy(RD_ogi),
        site_queue=[],
    )
    
    periodic_survey = Dm.site_survey.SiteSurvey(
        timeobj,
        survey_interval=copy.deepcopy(survey_interval_periodic),
        sites_per_day=copy.deepcopy(survey_sites_periodic),
        site_cost=copy.deepcopy(survey_cost_periodic),
        detection_variables={'flux': 'mean'},
        detection_probability_points=copy.deepcopy(periodic_detection_points),
        detection_probabilities=copy.deepcopy(periodic_detection_probabilities),
        dispatch_object=copy.deepcopy(ogi_called_survey),
        site_queue=[],
        ophrs=copy.deepcopy(survey_ophrs_periodic)
    )

    large_event_survey = Dm.site_survey.SiteSurvey(
        timeobj,
        survey_interval=copy.deepcopy(survey_interval_sat),
        sites_per_day=copy.deepcopy(survey_sites_sat),
        site_cost=copy.deepcopy(survey_cost_sat),
        detection_variables={'flux': 'mean'},
        detection_probability_points=copy.deepcopy(sat_detection_points),
        detection_probabilities=copy.deepcopy(sat_detection_probabilities),
        dispatch_object=copy.deepcopy(ogi_called_survey_sat),
        site_queue=[],
        ophrs=copy.deepcopy(survey_ophrs_sat)
    )

    #### 
    
    return ogi_quarterly, ogi_called_survey, ogi_called_survey_sat, ogi_annual_survey, periodic_survey, large_event_survey, RD_ogi


def define_ldar_programs(gas_field, ogi_quarterly, ogi_annual_survey, periodic_survey, large_event_survey, RD_ogi): ### RS Edits (5/18/22)
    """
    Define LDAR programs using the detection and repair methods defined previously
    :param gas_field: Emission simulation settings
    :param ogi_survery:  a quarterly OGI survey. 
    :param periodic_survey_LDAR: a periodic technology which calls an OGI followup
    :param periodic_survey_annual_LDAR: a periodic technology which calls OGI followup, and includes a seperate annual OGI survey. 
    :param periodic_survey_satellite_LDAR: a periodic technology which calls OGI followup, layered with a satellite survey which calls OGI survey.
    :param periodic_survey_satellite_annual_LDAR: periodic technology with OGI followup, satellite survey with OGI followup, and a seperate annual OGI survey. 
    :return ldar_dict: A dict of LDAR programs to be simulated
    """

    # Define LDAR programs

    #Okay, trying something from the RTI code...I think you can only call each detection method once without doing a copy function. 
    #this is so dumb. 
    ogi_quarterly.dispatch_object = copy.deepcopy(RD_ogi)
    # periodic_survey_do = copy.deepcopy(ogi_called_survey)
    periodic_survey_annual = copy.deepcopy(periodic_survey)
    periodic_survey_annual.dispatch_object = copy.deepcopy(ogi_called_survey)
    # periodic_survey_annual_do = copy.deepcopy(ogi_called_survey)
    periodic_satellite = copy.deepcopy(periodic_survey)
    periodic_satellite.dispatch_object = copy.deepcopy(ogi_called_survey)
    # periodic_satellite_do = copy.deepcopy(ogi_called_survey)
    periodic_satellite_sat = copy.deepcopy(large_event_survey)
    periodic_satellite_sat.dispatch_object = copy.deepcopy(ogi_called_survey_sat)
    # periodic_satellite_sat_do = copy.deepcopy(ogi_called_survey_sat)
    periodic_satellite_annual = copy.deepcopy(periodic_survey)
    periodic_satellite_annual.dispatch_object = copy.deepcopy(ogi_called_survey)
    # periodic_satellite_annual_do = copy.deepcopy(ogi_called_survey)
    periodic_satellite_sat_annual = copy.deepcopy(large_event_survey)
    periodic_satellite_sat_annual.dispatch_object = copy.deepcopy(ogi_called_survey_sat)
    # periodic_satellite_sat_annual_do = copy.deepcopy(ogi_called_survey_sat)


    #quarterly OGI 
    ogi_survey = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), {'ogi': ogi_quarterly},
    )
    # periodic survey no annual ogi
    tech_dict = {
        'periodic': periodic_survey,
        'called_ogi': periodic_survey.dispatch_object
    }
    periodic_survey_LDAR = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), tech_dict,
    )

    # periodic survey with annual ogi
    tech_dict = {
        'periodic': periodic_survey_annual,
        'called_ogi': periodic_survey_annual.dispatch_object,
        'annual_ogi': copy.deepcopy(ogi_annual_survey)
    }
    periodic_survey_annual_LDAR = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), tech_dict,
    )

    # periodic survey with Satellites, no annual ogi
    tech_dict = {
        'periodic': periodic_satellite,
        'called_ogi': periodic_satellite.dispatch_object,
        'satellite': periodic_satellite_sat,
        'satellite_ogi': periodic_satellite_sat.dispatch_object,
    }
    periodic_survey_satellite_LDAR = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), tech_dict,
    )

    # periodic survey with Satellites and annual ogi
    tech_dict = {
        'periodic': periodic_satellite_annual,
        'called_ogi': periodic_satellite_annual.dispatch_object,
        'satellite': periodic_satellite_sat_annual,
        'satellite_ogi': periodic_satellite_sat_annual.dispatch_object,
        'annual_ogi': copy.deepcopy(ogi_annual_survey)
    }
    periodic_survey_satellite_annual_LDAR = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), tech_dict,
    )


    # Combine All programs into a directory: 
    ldar_dict = {
        'quarterly_ogi': ogi_survey,
        'periodic': periodic_survey_LDAR,
        'periodic_annual': periodic_survey_annual_LDAR,
        'satellite' : periodic_survey_satellite_LDAR,
        'satellite_annual' : periodic_survey_satellite_annual_LDAR
    }
    ###
    
    return ldar_dict


#Setting up the looped verision of this code to run through different iterations, Creating and writing to new subfolders. 
for iter in cadence_values:
    #set the cadendce in days:
    survey_interval_periodic = cadence_values[iter]
    #and the folder name:
    write_folder = write_out_location + '/Loop_'+ str(Set_periodic_threshold)+ '_kg_' + str(iter) + '_' + str(survey_interval_periodic)

    os.mkdir(write_folder)

    for ind in range(n_montecarlo):
        print('Iteration number: {:0.0f}'.format(ind))
        comp_leak, comp_tank, comp_super = define_emitters()
        site_dict = define_sites(comp_leak, comp_tank, comp_super, mpStr)
        timeobj = define_time_settings()
        gas_field = define_gas_field(timeobj, site_dict)
        ogi_quarterly, ogi_called_survey, ogi_called_survey_sat, ogi_annual_survey, periodic_survey, large_event_survey, RD_ogi = define_detection_methods(timeobj) 
        ldar_dict = define_ldar_programs(gas_field, ogi_quarterly, ogi_annual_survey, periodic_survey, large_event_survey, RD_ogi)
        scenario = sc.Scenario(time=timeobj, gas_field=gas_field, ldar_program_dict=ldar_dict)
        scenario.run(dir_out=write_folder, display_status=False, save_method='json')

b = time.time()
print("run time {:0.2f} seconds".format(b - a))

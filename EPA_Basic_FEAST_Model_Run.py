"""
The Basic EPA Re-consideration Feast Modeling Script. 
This builds from the example script from RTI and the FEAST example run file. 

Running: Python 3.12.4 

V1.0 Date: 2025-06

Updates Tracking List: 
(add updates here)
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
write_out_location = 'EPA_Trial_Run/run_results_loop_test'

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
comp_emissions_start_leaks = (0.5/96 + 0.0000001) #Fraction of components expected to be emissting at the beginnging of the simulation
comp_emissions_rate_leaks = (0.005 / 365) # number of new emissions per component per day

#Tank Leaks (midsized low pressure equiptment)
emission_distribution_tanks = 'EPA_Trial_Run/Setup_Files/Comp_Tanks.p'
comp_emissions_start_tanks = (3/96 + 0.0000001) #Fraction of components expected to be emissting at the beginnging of the simulation
comp_emissions_rate_tanks = (0.025 / 365) # number of new emissions per component per day

#Super Emitters 
emission_distribution_super = 'EPA_Trial_Run/Setup_Files/Comp_Super.p'
comp_emissions_start_super = (0.5/96 + 0.0000001) #Fraction of components expected to be emissting at the beginnging of the simulation
comp_emissions_rate_super = (0.005 / 365) # number of new emissions per component per day

#TECHNOLOGY SETUP 
# All technologies have: detection_variables={'flux': 'mean'}, and site_queue = [] 
#Repair Delays
OGI_rep_delay = 30 #OGI repair delay in days
survey_rep_delay = 35 #Periodic survey repair delay in days
sat_rep_delay = 45 #Satellite survey repair delay in days 

#OGI Settings - General
survey_speed_comp_ogi = 150 #Survey speed in components/hour 
suvey_labor_cost_ogi = 100 #survey cost USD/HR 
survey_ophrs_ogi = {'begin': 8, 'end': 17}
ogi_detection_points= np.array([0.0042, 0.0063, 0.0083, 0.012, 0.0166]) #Detection distribution, g/s
ogi_detection_probabilities= np.array([0, 0.25, 0.5, 0.75, 1]) #Detecion Distribution, probabilities 

#Periodic Survey Settings - General 
# survey_interval_periodic = 30 #return period for the tech in days
survey_sites_periodic = 200 #how many sites are observed each survey day 
survey_cost_periodic = 100 #USD/Site cost for the survey
survey_ophrs_periodic = {'begin': 8, 'end': 17}
periodic_detection_points = np.array([0.14, 0.21, 0.28, 0.42, 0.56]) #detection Distribution, g/s
periodic_detection_probabilities = np.array([0, 0.25, 0.5, 0.75, 1]) #Detecion Distribution, probabilities

#Periodic Survey Settings - Large Event (satellite proxy) 
survey_interval_sat =180 #return period for the tech in days
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
        emission_data_path= emission_distribution_leaks,
        emission_per_comp= comp_emissions_start_leaks,  # Fraction of components expected to be emitting at the beginning of the simulation.
        emission_production_rate= comp_emissions_rate_leaks,  # number of new emissions per component per day
        repair_cost_path= repair_cost_distribution,
        base_reparable=True,
        null_repair_rate = 0.00000
    )
    
    # Generates reparable tank emissions
    comp_tank = feast.EmissionSimModules.infrastructure_classes.Component(
        name='Comp_Tanks',
        emission_data_path= emission_distribution_tanks,
        emission_per_comp= comp_emissions_start_tanks,  # Fraction of components expected to be emitting at the beginning of the simulation.
        emission_production_rate= comp_emissions_rate_tanks,  # number of new emissions per component per day
        repair_cost_path= repair_cost_distribution,
        base_reparable= True,
        null_repair_rate = 0.00000
    )

    # Generates reparable tank emissions
    comp_super = feast.EmissionSimModules.infrastructure_classes.Component(
        name='Comp_Super',
        emission_data_path= emission_distribution_super,
        emission_per_comp= comp_emissions_start_super,  # Fraction of components expected to be emitting at the beginning of the simulation.
        emission_production_rate= comp_emissions_rate_super,  # number of new emissions per component per day
        repair_cost_path= repair_cost_distribution,
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
    :return ogi: A component survey method representing OGI with periodic surveys
    :return ogi_no_survey: A component survey method representing OGI deployed by a site-level detection method
    :return plane: A site survey method representing a plane based detection program with periodic surveys
    :return cont_monitor: A site monitor method representing continuous monitors deployed at a site
    :return rep0: A repair method with 0 delay between detection and repair
    :return rep7: A repair method with a delay of 7 days between detection and repair
    """
    #Setting the return delays: 
    RD_ogi = Dm.repair.Repair(repair_delay = OGI_rep_delay)
    RD_survey = Dm.repair.Repair(repair_delay = survey_rep_delay)
    RD_satellite = Dm.repair.Repair(repair_delay = sat_rep_delay)


    
    ogi_quarterly = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=91,
        survey_speed= survey_speed_comp_ogi,
        ophrs= survey_ophrs_ogi,
        labor= suvey_labor_cost_ogi,
        detection_variables={'flux': 'mean'},
        detection_probability_points= ogi_detection_points,
        detection_probabilities= ogi_detection_probabilities,
        dispatch_object= RD_ogi,
        site_queue=[],
    )
    ogi_called_survey = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=None,
        survey_speed=survey_speed_comp_ogi,
        ophrs=survey_ophrs_ogi,
        labor=suvey_labor_cost_ogi,
        detection_variables={'flux': 'mean'},
        detection_probability_points=ogi_detection_points,
        detection_probabilities= ogi_detection_probabilities,
        dispatch_object= RD_survey,
        site_queue=[],
    )
    ogi_called_survey_sat = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=None,
        survey_speed=survey_speed_comp_ogi,
        ophrs=survey_ophrs_ogi,
        labor=suvey_labor_cost_ogi,
        detection_variables={'flux': 'mean'},
        detection_probability_points=ogi_detection_points,
        detection_probabilities= ogi_detection_probabilities,
        dispatch_object= RD_satellite,
        site_queue=[],
    )
    ogi_annual_survey = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=365,
        survey_speed=survey_speed_comp_ogi,
        ophrs=survey_ophrs_ogi,
        labor=suvey_labor_cost_ogi,
        detection_variables={'flux': 'mean'},
        detection_probability_points=ogi_detection_points,
        detection_probabilities= ogi_detection_probabilities,
        dispatch_object= RD_ogi,
        site_queue=[],
    )
    
    periodic_survey = Dm.site_survey.SiteSurvey(
        timeobj,
        survey_interval=survey_interval_periodic,
        sites_per_day=survey_sites_periodic,
        site_cost=survey_cost_periodic,
        detection_variables={'flux': 'mean'},
        detection_probability_points=periodic_detection_points,
        detection_probabilities=periodic_detection_probabilities,
        dispatch_object=ogi_called_survey,
        site_queue=[],
        ophrs=survey_ophrs_periodic
    )

    large_event_survey = Dm.site_survey.SiteSurvey(
        timeobj,
        survey_interval=survey_interval_sat,
        sites_per_day=survey_sites_sat,
        site_cost=survey_cost_sat,
        detection_variables={'flux': 'mean'},
        detection_probability_points=sat_detection_points,
        detection_probabilities=sat_detection_probabilities,
        dispatch_object=ogi_called_survey_sat,
        site_queue=[],
        ophrs=survey_ophrs_sat
    )

    #### 
    
    return ogi_quarterly, ogi_called_survey, ogi_called_survey_sat, ogi_annual_survey, periodic_survey, large_event_survey


def define_ldar_programs(gas_field, ogi_quarterly, ogi_annual_survey, periodic_survey, large_event_survey ): ### RS Edits (5/18/22)
    """
    Define LDAR programs using the detection and repair methods defined previously
    :param gas_field: Emission simulation settings
    :param ogi:  component survey method representing OGI with periodic surveys
    :param ogi_no_survey: A component survey method representing OGI deployed by a site-level detection method
    :param plane_survey: A site survey method representing a plane based detection program with periodic surveys
    :param cont_monitor: A site monitor method representing continuous monitors deployed at a site
    :param rep0: A repair method with 0 delay between detection and repair
    :param rep7: A repair method with a delay of 7 days between detection and repair
    :return ldar_dict: A dict of LDAR programs to be simulated
    """
    # Add dispatch methods and site specific conditions to detection methods
    # Good practice to use copies so that LDAR programs do not interfere with eachother in the simulation
    # ogi.dispatch_object = copy.deepcopy(rep0)
    # ogi_no_survey.dispatch_object = copy.deepcopy(rep0)
    # plane_ogi = copy.deepcopy(ogi_no_survey)
    # plane_ogi2 = copy.deepcopy(ogi_no_survey2)
    # plane_survey.dispatch_object = plane_ogi
    # plane_survey2.dispatch_object = plane_ogi2
 
    # Define LDAR programs

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
        'periodic': periodic_survey,
        'called_ogi': periodic_survey.dispatch_object,
        'annual_ogi': ogi_annual_survey
    }
    periodic_survey_annual_LDAR = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), tech_dict,
    )

    # periodic survey with Satellites and annual ogi
    tech_dict = {
        'periodic': periodic_survey,
        'called_ogi': periodic_survey.dispatch_object,
        'satellite': large_event_survey,
        'satellite_ogi': large_event_survey.dispatch_object,
        'annual_ogi': ogi_annual_survey
    }
    periodic_survey_satellite_annual_LDAR = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), tech_dict,
    )

    # periodic survey with Satellites, no annual ogi
    tech_dict = {
        'periodic': periodic_survey,
        'called_ogi': periodic_survey.dispatch_object,
        'satellite': large_event_survey,
        'satellite_ogi': large_event_survey.dispatch_object,
    }
    periodic_survey_satellite_LDAR = Dm.ldar_program.LDARProgram(
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
    write_folder = write_out_location + '/Loop_2kg_' + str(iter)

    os.mkdir(write_folder)

    for ind in range(n_montecarlo):
        print('Iteration number: {:0.0f}'.format(ind))
        comp_leak, comp_tank, comp_super = define_emitters()
        site_dict = define_sites(comp_leak, comp_tank, comp_super, mpStr)
        timeobj = define_time_settings()
        gas_field = define_gas_field(timeobj, site_dict)
        ogi_quarterly, ogi_called_survey, ogi_called_survey_sat, ogi_annual_survey, periodic_survey, large_event_survey = define_detection_methods(timeobj) 
        ldar_dict = define_ldar_programs(gas_field, ogi_quarterly, ogi_annual_survey, periodic_survey, large_event_survey)
        scenario = sc.Scenario(time=timeobj, gas_field=gas_field, ldar_program_dict=ldar_dict)
        scenario.run(dir_out=write_folder, display_status=True, save_method='json')

b = time.time()
print("run time {:0.2f} seconds".format(b - a))

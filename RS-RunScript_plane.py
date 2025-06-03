"""
I'm modifying this script from what RTI provided; this will pull from the same underlying emissions distributions, and run the same LDAR programs. 
The main goal here is to make sure that I've gotten eveything set up correctly and we're generating the model runs. 
"""
import numpy as np
import copy
import feast.EmissionSimModules.infrastructure_classes
import feast.EmissionSimModules.simulation_classes as sc
from feast.EmissionSimModules.infrastructure_classes import Component
import feast.DetectionModules as Dm
import pickle
import time
import json
import os
from scipy.special import erf

###############################################################################
# USER INPUT
mpStr = 'Model Plant 4'
n_montecarlo = 500

###############################################################################

# The random seed below can be un-commented to generate reproducible realizations.
# np.random.seed(0)

a = time.time()
rsc_path = os.getcwd()

def define_emitters():
    """
    Defines all emitters to be used in the simulation using the Component class
    :return comp_fug: source of fugitive emissions
    :return misc_vent: source of miscelaneous vents
    :return plunger: source of unloading emissions due to wells with plungers
    :return noplunger: source of unloading emission due to wells without plungers
    """
    # Generates reparable fugitive emissions
    comp_leak = feast.EmissionSimModules.infrastructure_classes.Component(
        name='Comp_Leaks',
        emission_data_path=os.path.join(rsc_path, 'EPA_Trial_Run', 'Setup_Files','Comp_Leaks.p'),
        emission_per_comp=(0.5/96 + 0.0000001),  # Fraction of components expected to be emitting at the beginning of the simulation.
        emission_production_rate=(0.005 / 365),  # number of new emissions per component per day
        repair_cost_path='EPA_Trial_Run/Setup_Files/fernandez_leak_repair_costs_2006.p',
        base_reparable=True,
        null_repair_rate = 0.00000
    )
    
    # Generates reparable tank emissions
    comp_tank = feast.EmissionSimModules.infrastructure_classes.Component(
        name='Comp_Tanks',
        emission_data_path=os.path.join(rsc_path, 'EPA_Trial_Run', 'Setup_Files','Comp_Tanks.p'),
        emission_per_comp=(3/96 + 0.0000001),  # Fraction of components expected to be emitting at the beginning of the simulation.
        emission_production_rate=(0.025 / 365),  # number of new emissions per component per day
        repair_cost_path='EPA_Trial_Run/Setup_Files/fernandez_leak_repair_costs_2006.p',
        base_reparable=True,
        null_repair_rate = 0.00000
    )

    # Generates reparable tank emissions
    comp_super = feast.EmissionSimModules.infrastructure_classes.Component(
        name='Comp_Super',
        emission_data_path=os.path.join(rsc_path, 'EPA_Trial_Run', 'Setup_Files','Comp_Super.p'),
        emission_per_comp=(0.5/96 + 0.0000001),  # Fraction of components expected to be emitting at the beginning of the simulation.
        emission_production_rate=(0.005 / 365),  # number of new emissions per component per day
        repair_cost_path='EPA_Trial_Run/Setup_Files/fernandez_leak_repair_costs_2006.p',
        base_reparable=True,
        null_repair_rate = 0.00000
    )

    return comp_leak, comp_tank, comp_super


def define_sites(comp_leak, comp_tank, comp_super, modelPlantString):
    """
    Defines all sites to be used in the simulation. Sites consist of a collection of Component objects

    """
    
    nSites = 20
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
    return feast.EmissionSimModules.simulation_classes.Time(delta_t=1, end_time=365*5)


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
    gas_field.met_data_path = 'ExampleData/TMY-DataExample.csv'
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
    points = np.logspace(-3, 1, 100)  # emission rates
    rep0 = Dm.repair.Repair(repair_delay=30)
    rep7 = Dm.repair.Repair(repair_delay=30)
    probs = 0.5 + 0.5 * np.array([erf((np.log(f) - np.log(0.02)) / (0.8 * np.sqrt(2))) for f
                                  in points])
    probs[0] = 0
    ogi = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=91,
        survey_speed=150,
        ophrs={'begin': 8, 'end': 17},
        labor=100,
        detection_variables={'flux': 'mean'},
        detection_probability_points=np.array([0.0042, 0.0063, 0.0083, 0.012, 0.0166]),
        detection_probabilities=np.array([0, 0.25, 0.5, 0.75, 1]),
        dispatch_object=rep0,
        site_queue=[],
    )
    ogi_no_survey = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=None,
        survey_speed=150,
        ophrs={'begin': 8, 'end': 17},
        labor=100,
        detection_variables={'flux': 'mean'},
        detection_probability_points=np.array([0.0042, 0.0063, 0.0083, 0.012, 0.0166]),
        detection_probabilities=np.array([0, 0.25, 0.5, 0.75, 1]),
        dispatch_object=copy.copy(rep7),
        site_queue=[],
    )
    ogi_no_survey2 = Dm.comp_survey.CompSurvey(
        timeobj,
        survey_interval=365,
        survey_speed=150,
        ophrs={'begin': 8, 'end': 17},
        labor=100,
        detection_variables={'flux': 'mean'},
        detection_probability_points=np.array([0.0042, 0.0063, 0.0083, 0.012, 0.0166]),
        detection_probabilities=np.array([0, 0.25, 0.5, 0.75, 1]),
        dispatch_object=copy.copy(rep7),
        site_queue=[],
    )
    points = np.logspace(-3, 1, 100)
    # 0.474
    probs = 0.5 + 0.5 * np.array([erf((np.log(f) - np.log(1.5)) / (1.36 * np.sqrt(2))) for f
                                  in points])
    probs[0] = 0
    plane_survey = Dm.site_survey.SiteSurvey(
        timeobj,
        survey_interval=30,
        sites_per_day=200,
        site_cost=100,
        detection_variables={'flux': 'mean'},
        detection_probability_points=np.array([0.14, 0.21, 0.28, 0.42, 0.56]),
        detection_probabilities=np.array([0, 0.25, 0.5, 0.75, 1]),
        dispatch_object=ogi_no_survey,
        site_queue=[],
        ophrs={'begin': 8, 'end': 17}
    )
    #### RS EDIT (5/18/22)
    plane_survey2 = Dm.site_survey.SiteSurvey(
        timeobj,              
        survey_interval=30,
        sites_per_day=200,
        site_cost=100,
        detection_variables={'flux': 'mean'},
        detection_probability_points=np.array([0.14, 0.21, 0.28, 0.42, 0.56]),
        detection_probabilities=np.array([0, 0.25, 0.5, 0.75, 1]),
        dispatch_object=ogi_no_survey2,
        site_queue=[],
        ophrs={'begin': 8, 'end': 17}
    )
    #### 
    cont_monitor = Dm.site_monitor.SiteMonitor(
        timeobj,
        time_to_detect_points=[[0.5, 1], [1.0, 1], [1.1, 1], [0.5, 5], [1.0, 5], [1.1, 5],
                               [0.5, 5.1], [1.0, 5.1], [1.1, 5.1]],
        time_to_detect_days=[np.inf, 1, 0, np.inf, 5, 0, np.inf, np.inf, np.inf],
        detection_variables={'flux': 'mean', 'wind speed': 'mean'},
        site_queue=list(range(gas_field.n_sites)),
        dispatch_object=copy.deepcopy(rep0),
        ophrs={'begin': 8, 'end': 17}
    )
    return ogi, ogi_no_survey, plane_survey, plane_survey2, cont_monitor, rep0, rep7, ogi_no_survey2 ### RS Edits (5/18/22)


def define_ldar_programs(gas_field, ogi, ogi_no_survey, plane_survey, plane_survey2, cont_monitor, rep0, rep7, ogi_no_survey2): ### RS Edits (5/18/22)
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
    ogi.dispatch_object = copy.deepcopy(rep0)
    ogi_no_survey.dispatch_object = copy.deepcopy(rep0)
    plane_ogi = copy.deepcopy(ogi_no_survey)
    plane_ogi2 = copy.deepcopy(ogi_no_survey2)
    plane_survey.dispatch_object = plane_ogi
    plane_survey2.dispatch_object = plane_ogi2
    cm_ogi = copy.deepcopy(ogi_no_survey)
    cont_monitor.dispatch_object = cm_ogi
    cont_monitor.site_queue = np.linspace(0, gas_field.n_sites - 1, gas_field.n_sites, dtype=int)
    cont_monitor.op_envelope = {
        'wind direction': {'class': 2,
                           'min': [[45, 225]] * gas_field.n_sites,
                           'max': [[135, 315]] * gas_field.n_sites}
    }
    # Define LDAR programs
    ogi_survey = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), {'ogi': ogi},
    )
    # tiered survey
    tech_dict = {
        'plane': plane_survey,
        'ogi': plane_ogi
    }
    plane_ogi_survey = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), tech_dict,
    )
    
    #### RS EDIT (5/18/22)
    # tiered survey 2
    tech_dict = {
        'plane2': plane_survey2,
        'ogi': plane_ogi2
    }
    plane_ogi_survey2 = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), tech_dict,
    )
    #### 
    
    # continuous monitor
    tech_dict = {
        'cm': cont_monitor,
        'ogi': cm_ogi
    }
    cm_ogi = Dm.ldar_program.LDARProgram(
        copy.deepcopy(gas_field), tech_dict,
    )
    
    #### RS EDIT (5/18/22): Added 'plane2' entry
    # All programs
    ldar_dict = {
        'cm': cm_ogi,
        'ogi': ogi_survey,
        'plane': plane_ogi_survey,
        'plane2': plane_ogi_survey2
    }
    ###
    
    return ldar_dict


for ind in range(n_montecarlo):
    print('Iteration number: {:0.0f}'.format(ind))
    comp_leak, comp_tank, comp_super = define_emitters()
    site_dict = define_sites(comp_leak, comp_tank, comp_super, mpStr)
    timeobj = define_time_settings()
    gas_field = define_gas_field(timeobj, site_dict)
    ogi, ogi_no_survey, plane_survey, plane_survey2, cont_monitor, rep0, rep7, ogi_no_survey2 = define_detection_methods(timeobj) #### RS EDIT (5/18/22)
    ldar_dict = define_ldar_programs(gas_field, ogi, ogi_no_survey, plane_survey, plane_survey2, cont_monitor, rep0, rep7, ogi_no_survey2) #### RS EDIT (5/18/22)
    scenario = sc.Scenario(time=timeobj, gas_field=gas_field, ldar_program_dict=ldar_dict)
    scenario.run(dir_out='EPA_Trial_Run/Run_Results', display_status=True, save_method='json')

b = time.time()
print("run time {:0.2f} seconds".format(b - a))

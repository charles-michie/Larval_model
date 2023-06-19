# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway
# 
##################################################################################################
# 
# Module to simulate behaviour of Plankton, including phytoplankton (plants) and zooplankton (animals)
#
# Developed by Simon WeppeMetOcean/MetService NZ based on several sources to fit requirements for 
# use in MetOceanTrack model developped by MetOcean/MetService NZ as part of the Moana project 
# (https://www.moanaproject.org/)
# 
##################################################################################################
# 
# Sources:
# 
# This module is based on several sources:
# a) the work and codes of Kristiansen,Romagnoni,Kvile
#   >> https://github.com/trondkr/KINO-ROMS/tree/master/Romagnoni-2019-OpenDrift/kino
#   >> https://github.com/trondkr/KINO-ROMS/blob/master/Romagnoni-2019-OpenDrift/kino/pelagicplankton.py
#
#   "This code simulates cod egg and larvae and was developed by Trond Kristiansen (me (at) trondkristiansen.com)
#   and Kristina Kvile (kristokv (at) gmail.com)""
# 
# b) Opendrift's pelagicegg module 
#   >> https://github.com/OpenDrift/opendrift/blob/master/opendrift/models/pelagicegg.py
# 
# c ) Some components of code are taken from 
#   >> https://github.com/metocean/ercore/tree/ercore_opensrc/ercore/lib
# 
##################################################################################################

import os
import numpy as np
from datetime import datetime, timedelta,timezone

from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray
from opendrift.elements import LagrangianArray
import logging; logger = logging.getLogger(__name__)


# Defining the PelagicPlankton element properties
class MScLarvae(Lagrangian3DArray):
    """Extending Lagrangian3DArray with specific properties for pelagic plankton

    """

    variables = Lagrangian3DArray.add_variables([
        ('diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.0014}), 
        ('neutral_buoyancy_salinity', {'dtype': np.float32,
                                      'units': '[]',
                                       'default': 31.25}), 
        ('age_seconds', {'dtype': np.float32,
                         'units': 's',
                         'default': 0.}),           
         ('light', {'dtype': np.float32,
                     'units': 'ugEm2',
                     'default': 0.}),
         ('survival', {'dtype': np.float32,
                          'units': '',
                          'default': 1.})])

class MScLarvaeDrift(OceanDrift):
    """Buoyant particle trajectory model based on the OpenDrift framework.

        Developed at MET Norway

        Generic module for particles that are subject to vertical turbulent
        mixing with the possibility for positive or negative buoyancy

        Particles could be e.g. oil droplets, plankton, or sediments

        Under construction.
    """

    ElementType = MScLarvae

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_wave_significant_height': {'fallback': 0},
        'sea_ice_area_fraction': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 100},
        'ocean_vertical_diffusivity': {'fallback': 0.02, 'profiles': True},
        'ocean_mixed_layer_thickness': {'fallback': 50},
        'sea_water_temperature': {'fallback': 10, 'profiles': True},
        'sea_water_salinity': {'fallback': 34, 'profiles': True},
        'surface_downward_x_stress': {'fallback': 0},
        'surface_downward_y_stress': {'fallback': 0},
        'turbulent_kinetic_energy': {'fallback': 0},
        'turbulent_generic_length_scale': {'fallback': 0},
        'upward_sea_water_velocity': {'fallback': 0},
        'ocean_mixed_layer_thickness': {'fallback': 50},
      }

    # The depth range (in m) which profiles shall cover
    required_profiles_z_range = [-200, 0] 

    required_profiles = ['sea_water_temperature',
                         'sea_water_salinity',
                         'ocean_vertical_diffusivity']

    # Default colors for plotting
    status_colors = {'initial': 'green', 'active': 'blue',
                     'settled_on_coast': 'red', 'retired': 'black', 
                     'settled_on_bottom': 'yellow'}

    def __init__(self, *args, **kwargs):

        # Calling general constructor of parent class
        super(MScLarvaeDrift, self).__init__(*args, **kwargs)

        # By default, eggs do not strand towards coastline
        # self.set_config('general:coastline_action', 'previous')
        self._set_config_default('general:coastline_action', 'previous')
        self._set_config_default('general:seafloor_action', 'lift_to_seafloor') #added 23/03/2022 

        # Vertical mixing is enabled by default
        # self.set_config('drift:vertical_mixing', True)
        self._set_config_default('drift:vertical_mixing', True)
        
        #################################################################################################
        # IBM specifications based on ERcore Plankton module:
        # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L25

        self._add_config({ 'biology:mortality_daily_rate': {'type': 'float', 'default': 0.05,'min': 0.0, 'max': 100.0, 'units': 'percentage of biomass dying per day',
                           'description': 'Mortality rate (percentage of biomass dying per day)',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:min_settlement_age_seconds': {'type': 'float', 'default': 0.00,'min': 0.0, 'max': 7.0e6, 'units': 'seconds',
                           'description': 'Minimum age before beaching can occur, in seconds',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:heavy_age_seconds': {'type': 'float', 'default': 0.00,'min': 0.0, 'max': 7.0e6, 'units': 'seconds',
                           'description': 'Age when larvae become heavy and go towards bottom fast',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:max_settlement_depth': {'type': 'float', 'default': -20,'min': -1000.0, 'max': 0, 'units': 'meters negative down',
                           'description': 'Maximum depth they can settle at',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:vertical_position_daytime': {'type': 'float', 'default': -5.00,'min': -1000.0, 'max':0.0, 'units': 'meters negative down',
                           'description': 'the depth a species is expected to inhabit during the day time, in meters, negative down',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:vertical_position_nighttime': {'type': 'float', 'default': -1.00,'min': -1000.0, 'max':0.0, 'units': 'meters negative down',
                           'description': 'the depth a species is expected to inhabit during the night time, in meters, negative down',
                           'level': self.CONFIG_LEVEL_BASIC}})
        # vertical_migration_speed_constant is the speed at which larvae will move to vertical_position_daytime or vertical_position_nighttime (depeding if it is day or nighttime)
        # if set to None the model will use the vertical velocity defined in update_terminal_velocity() (and so will not take into account vertical_position_daytime/vertical_position_nighttime)
        self._add_config({ 'biology:vertical_migration_speed_constant': {'type': 'float', 'default': None,'min': 0.0, 'max': 1.0e-3, 'units': 'm/s',
                           'description': 'Constant vertical migration rate (m/s), if None, use values from update_terminal_velocity()',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:temperature_min': {'type': 'float', 'default': None,'min': 0.0, 'max': 100.0, 'units': 'degrees Celsius',
                           'description': 'lower threshold temperature where a species population quickly declines to extinction in degrees Celsius',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:temperature_max': {'type': 'float', 'default': None,'min': 0.0, 'max': 100.0, 'units': 'degrees Celsius',
                           'description': 'upper threshold temperature where a species population quickly declines to extinction in degrees Celsius',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:temperature_tolerance': {'type': 'float', 'default': 1.0,'min': 0.0, 'max': 1.0, 'units': 'degrees Celsius',
                           'description': 'temperature tolerance before dying, in degrees Celsius',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:salinity_min': {'type': 'float', 'default': None,'min': 0.0, 'max': 100.0, 'units': 'ppt',
                           'description': 'lower threshold salinity where a species population quickly declines to extinction in ppt',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:salinity_max': {'type': 'float', 'default': None,'min': 0.0, 'max': 100.0, 'units': 'ppt',
                           'description': 'upper threshold salinity where a species population quickly declines to extinction in ppt',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:salinity_tolerance': {'type': 'float', 'default': 0.1,'min': 0.0, 'max': 1.0, 'units': 'ppt',
                           'description': 'salinity tolerance before dying, in ppt',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:heavy_terminal_velocity': {'type': 'float', 'default': None,'min': -5, 'max': 0, 'units': 'm/s',
                           'description': 'Terminal velocity of heavy stage',
                           'level': self.CONFIG_LEVEL_BASIC}})

    def update_terminal_velocity(self,Tprofiles=None, Sprofiles=None,
                                 z_index=None): 

        if self.get_config('biology:min_settlement_age_seconds') is not None: 
            
            younger = np.where(self.elements.age_seconds < self.get_config('biology:min_settlement_age_seconds'))
            heavy = np.where(self.elements.age_seconds >= self.get_config('biology:min_settlement_age_seconds'))
            
            younger_speed = 0.002
            heavy_speed = self.get_config('biology:heavy_terminal_velocity')
            
            #self.update_terminal_velocity_pelagicegg()
            self.elements.terminal_velocity[(younger)] = younger_speed
            self.elements.terminal_velocity[(heavy)] = heavy_speed
                 
            
         

    def update_terminal_velocity_pelagicegg(self, Tprofiles=None,
                                 Sprofiles=None, z_index=None):
        """Calculate terminal velocity for Pelagic Egg

            according to
            S. Sundby (1983): A one-dimensional model for the vertical
            distribution of pelagic fish eggs in the mixed layer
            Deep Sea Research (30) pp. 645-661

            Method copied from ibm.f90 module of LADIM:
            Vikebo, F., S. Sundby, B. Aadlandsvik and O. Otteraa (2007),
            Fish. Oceanogr. (16) pp. 216-228

            same as Opendrift's PelagicEggDrift model. This function is called in update()
        """ 
        g = 9.81  # ms-2

        # Pelagic Egg properties that determine buoyancy
        eggsize = self.elements.diameter  # 0.0014 for NEA Cod
        eggsalinity = self.elements.neutral_buoyancy_salinity
        # 31.25 for NEA Cod

        # prepare interpolation of temp, salt
        if not (Tprofiles is None and Sprofiles is None):
            if z_index is None:
                z_i = range(Tprofiles.shape[0])  # evtl. move out of loop
                # evtl. move out of loop
                z_index = interp1d(-self.environment_profiles['z'],
                                   z_i, bounds_error=False)
            zi = z_index(-self.elements.z)
            upper = np.maximum(np.floor(zi).astype(np.int), 0)
            lower = np.minimum(upper+1, Tprofiles.shape[0]-1)
            weight_upper = 1 - (zi - upper)

        # do interpolation of temp, salt if profiles were passed into
        # this function, if not, use reader by calling self.environment
        if Tprofiles is None:
            T0 = self.environment.sea_water_temperature
        else:
            T0 = Tprofiles[upper, range(Tprofiles.shape[1])] * \
                weight_upper + \
                Tprofiles[lower, range(Tprofiles.shape[1])] * \
                (1-weight_upper)
        if Sprofiles is None:
            S0 = self.environment.sea_water_salinity
        else:
            S0 = Sprofiles[upper, range(Sprofiles.shape[1])] * \
                weight_upper + \
                Sprofiles[lower, range(Sprofiles.shape[1])] * \
                (1-weight_upper)

        # The density difference bettwen a pelagic egg and the ambient water
        # is regulated by their salinity difference through the
        # equation of state for sea water.
        # The Egg has the same temperature as the ambient water and its
        # salinity is regulated by osmosis through the egg shell.
        DENSw = self.sea_water_density(T=T0, S=S0)
        DENSegg = self.sea_water_density(T=T0, S=eggsalinity)
        dr = DENSw-DENSegg  # density difference

        # water viscosity
        my_w = 0.001*(1.7915 - 0.0538*T0 + 0.007*(T0**(2.0)) - 0.0023*S0)
        # ~0.0014 kg m-1 s-1

        # terminal velocity for low Reynolds numbers
        W = (1.0/my_w)*(1.0/18.0)*g*eggsize**2 * dr

        # check if we are in a Reynolds regime where Re > 0.5
        highRe = np.where(W*1000*eggsize/my_w > 0.5)

        # Use empirical equations for terminal velocity in
        # high Reynolds numbers.
        # Empirical equations have length units in cm!
        my_w = 0.01854 * np.exp(-0.02783 * T0)  # in cm2/s
        d0 = (eggsize * 100) - 0.4 * \
            (9.0 * my_w**2 / (100 * g) * DENSw / dr)**(1.0 / 3.0)  # cm
        W2 = 19.0*d0*(0.001*dr)**(2.0/3.0)*(my_w*0.001*DENSw)**(-1.0/3.0)
        # cm/s
        W2 = W2/100.  # back to m/s

        W[highRe] = W2[highRe]
        self.elements.terminal_velocity = W

    ##################################################################################################
    # IBM-specific routines
    ##################################################################################################
    # >> reproduce behaviours included in Plankton class
    #    in https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py
    ##################################################################################################
    
    def calculateMaxSunLight(self):
        # Calculates the max sun radiation at given positions and dates (and returns zero for night time)
        # 
        # The method is using the third party library PySolar : https://pysolar.readthedocs.io/en/latest/#
        # 
        # some other available options:
        # https://pypi.org/project/solarpy/
        # https://github.com/trondkr/pyibm/blob/master/light.py
        # use calclight from Kino Module here  : https://github.com/trondkr/KINO-ROMS/tree/master/Romagnoni-2019-OpenDrift/kino
        # ERcore : dawn and sunset times : https://github.com/metocean/ercore/blob/ercore_opensrc/ercore/lib/suncalc.py
        # https://nasa-develop.github.io/dnppy/modules/solar.html#examples
        # 
        from pysolar import solar
        date = self.time
        date = date.replace(tzinfo=timezone.utc) # make the datetime object aware of timezone, set to UTC
        logger.debug('Assuming UTC time for solar calculations')
        # longitude convention in pysolar, consistent with Opendrift : negative reckoning west from prime meridian in Greenwich, England
        # the particle longitude should be converted to the convention [-180,180] if that is not the case
        sun_altitude = solar.get_altitude(self.elements.lat, self.elements.lon, date) # get sun altitude in degrees ** 
        sun_azimut = solar.get_azimuth(self.elements.lat, self.elements.lon, date) # get sun azimuth in degrees
        sun_radiation = np.zeros(len(sun_azimut))
         # not ideal get_radiation_direct doesnt accept arrays...
        for elem_i,alt in enumerate(sun_altitude):
            sun_radiation[elem_i] = solar.radiation.get_radiation_direct(date, alt)  # watts per square meter [W/m2] for that time of day
        self.elements.light = sun_radiation * 4.6 #Converted from W/m2 to umol/m2/s-1"" - 1 W/m2 ≈ 4.6 μmole.m2/s
        logger.debug('Solar radiation from %s to %s [W/m2]' % (sun_radiation.min(), sun_radiation.max() ) )
        # print(np.min(sun_radiation))
        # print(date)
    
    def plankton_development(self):

        self.update_survival() # mortality based on user-input mortality rate
        self.update_weight_temperature() # # mortality based on "liveable" temperature range
        self.update_weight_salinity() # # mortality based on "liveable" salinity range

    def update_survival(self):
        # update survval fraction based on mortality rate in [day-1]
        mortality_daily_rate = self.get_config('biology:mortality_daily_rate')
        # update survival fraction, accounting for mortality rate over that timestep
        fraction_died = self.elements.survival * mortality_daily_rate * (self.time_step.total_seconds()/(86400))
        self.elements.survival -= fraction_died
        

    def update_weight_temperature(self):
        #update particle survival fraction based on temperature, if a temperature range was input
        temp_tol = self.get_config('biology:temperature_tolerance')
        temp_min = self.get_config('biology:temperature_min')
        temp_max = self.get_config('biology:temperature_max')
        temp_xy = self.environment.sea_water_temperature # at particle positions

        if (temp_min is not None) or (temp_max is not None) :
            if temp_max is not None :
                m=(temp_xy-temp_max+temp_tol)/temp_tol # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L60
                if (m>0).any():
                  logger.debug('Maximum temperature reached for %s particles' % np.sum(m>0))
                self.elements.survival -= np.maximum(np.minimum(m,1),0)*self.elements.survival # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L62
            if temp_min is not None :
                m=(temp_min+temp_tol-temp_xy)/temp_tol # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L64
                # not sure what was intended behavior for 'm' here, and what temp_tol really means.
                # as is, it means all particle start to die when temp_xy goes below [temp_min+temp_tol], and are all dead when temp_xy reaches [temp_min]
                # >> m=(temp_min+temp_tol-temp_xy)/temp_tol
                # 
                # other option could be that particle start to decay at temp_min, and are all dead when reaching [temp_min-temp_tol]
                # >> m=(temp_min-temp_xy)/temp_tol

                if (m>0).any():
                  logger.debug('Minimum temperature reached for %s particles' % np.sum(m>0))
                self.elements.survival -= np.maximum(np.minimum(m,1),0)*self.elements.survival # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L65

        # print('TEMP')
        # print(self.elements.survival)
        # import pdb;pdb.set_trace()

    def update_weight_salinity(self):
        #update particle survival fraction based on salinity, if a salinity range was input
        salt_tol = self.get_config('biology:salinity_tolerance')
        salt_min = self.get_config('biology:salinity_min')
        salt_max = self.get_config('biology:salinity_max')
        salt_xy = self.environment.sea_water_salinity # at particle positions

        if (salt_min is not None) or (salt_max is not None) :
            if salt_max is not None :
                m=(salt_xy-salt_max+salt_tol)/salt_tol # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L60
                if (m>0).any():
                  logger.debug('Maximum salinity reached for %s particles' % np.sum(m>0))
                self.elements.survival -= np.maximum(np.minimum(m,1),0)*self.elements.survival # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L73
            if salt_min is not None :
                m=(salt_min+salt_tol-salt_xy)/salt_tol # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L64
                if (m>0).any():
                  logger.debug('Minimum salinity reached for %s particles' % np.sum(m>0))
                self.elements.survival -= np.maximum(np.minimum(m,1),0)*self.elements.survival # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L77
        # print('SALT')
        # print(self.elements.survival)
        # import pdb;pdb.set_trace()

    def update_terminal_velocity_constant(self):
        # modifies the same variable than update_terminal_velocity(), self.elements.terminal_velocity = W, but using a different algorithm.
        # Larvae are assumed to move to daytime or nighttime vertical positions in the water column, at a constant rate
        #
        # the actual settling is taken care of in vertical_mixing() or vertical_buoyancy() (i.e. from OceanDrift methods)
        self.calculateMaxSunLight() # compute solar radiation at particle positions (using PySolar)
        # it is expected that larve will go down during day time and up during night time but that is not fixed in the code. 
        # Particles will simply go towards the daytime or nighttime poistions.
        # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L80
        vertical_velocity = np.abs(self.get_config('biology:vertical_migration_speed_constant'))  # magnitude in m/s 
        z_day = self.get_config('biology:vertical_position_daytime')    #  the depth a species is expected to inhabit during the day time, in meters, negative down') #
        z_night =self.get_config('biology:vertical_position_nighttime') # 'the depth a species is expected to inhabit during the night time, in meters, negative down') #
        ind_day = np.where(self.elements.light>0)
        ind_night = np.where(self.elements.light==0)
        logger.debug('Using constant migration rate (%s m/s) towards day and night time positions' % (vertical_velocity) )
        logger.debug('%s particles in day time' % (len(ind_day[0])))
        logger.debug('%s particles in night time' % (len(ind_night[0])))
        # for particles in daytime : particles below the daytime position need to go up while particles above the daytime position need to go down
        # (same for for particles in nightime)
        # Note : depth convention is neagtive down in Opendrift
        # 
        # e.g. z=-5, z_day = -3, below daytime position,  need to go up (terminal_velocity>0) 
        #      diff = (z - z_day) = -2, so w = - np.sign(diff) * vertical_velocity
        self.elements.terminal_velocity[ind_day] = - np.sign(self.elements.z[ind_day] - z_day) * vertical_velocity
        self.elements.terminal_velocity[ind_night] = - np.sign(self.elements.z[ind_night] - z_night) * vertical_velocity
        # print(self.elements.z)

    def spawn(self):
        pass
        # particles change types after a certain time
        # need more info to implement
        # this could call a new seed_elements() method (that may or may not need to be changed relative to the one in basemodel.py) or 
        # simply change some parameters of existing particles at a certan time...
        # 

    #################################################################################################################################
    # SEAFLOOR AND SEABED INTERACTION
    #################################################################################################################################
    def interact_with_seafloor(self):
        """Seafloor interaction according to configuration setting"""
        # 
        # This function will overloads the version in basemodel.py
        if self.num_elements_active() == 0:
            return
        if 'sea_floor_depth_below_sea_level' not in self.priority_list:
            return
        sea_floor_depth = self.sea_floor_depth()
        below = np.where(self.elements.z < -sea_floor_depth)[0]
        if len(below) == 0:
                logger.debug('No elements hit seafloor.')
                return

        below_and_older = np.logical_and(self.elements.z < -sea_floor_depth, 
            self.elements.age_seconds >= self.get_config('biology:min_settlement_age_seconds'))
        below_and_younger = np.logical_and(self.elements.z < -sea_floor_depth, 
            self.elements.age_seconds < self.get_config('biology:min_settlement_age_seconds'))
        
        # Move all elements younger back to seafloor 
        # (could rather be moved back to previous if relevant? )
        self.elements.z[np.where(below_and_younger)] = -sea_floor_depth[np.where(below_and_younger)] + 0.5

        # deactivate elements that were both below and older
        
        max_settlement_depth = self.get_config('biology:max_settlement_depth')
        self.deactivate_elements(np.logical_and(below_and_older, self.elements.z > max_settlement_depth) ,reason='settled_on_bottom')

        logger.debug('%s elements hit seafloor, %s were older than %s sec. and deactivated, %s were lifted back to seafloor' \
            % (len(below),len(below_and_older),self.get_config('biology:min_settlement_age_seconds'),len(below_and_younger)))    

        # original code
        # https://github.com/OpenDrift/opendrift/blob/4dbd9a607fe23e64dcbf9fd05905af8713dc74d1/opendrift/models/basemodel.py#L613 

    def interact_with_coastline(self,final = False): 
        """Coastline interaction according to configuration setting
           (overloads the interact_with_coastline() from basemodel.py)
           
           The method checks for age of particles that intersected coastlines:
             if age_particle < min_settlement_age_seconds : move larvaes back to previous wet position
             if age_particle > min_settlement_age_seconds : larvaes become stranded and will be deactivated.
        """
        i = self.get_config('general:coastline_action') # will always be 'previous'

        if not hasattr(self.environment, 'land_binary_mask'):
            return

        if final is True:  # Get land_binary_mask for final location
            en, en_prof, missing = \
                self.get_environment(['land_binary_mask'],
                                     self.time,
                                     self.elements.lon,
                                     self.elements.lat,
                                     self.elements.z,
                                     None)
            self.environment.land_binary_mask = en.land_binary_mask

        # if i == 'previous':  # Go back to previous position (in water)
        # previous_position_if = self.previous_position_if()
        if self.newly_seeded_IDs is not None:
            self.deactivate_elements(
                (self.environment.land_binary_mask == 1) &
                (self.elements.ID >= self.newly_seeded_IDs[0]),
                reason='seeded_on_land')
        on_land = np.where(self.environment.land_binary_mask == 1)[0]

        # if previous_position_if is not None:
        #     self.deactivate_elements((previous_position_if*1 == 1) & (
        #                      self.environment.land_binary_mask == 0),
        #                          reason='seeded_at_nodata_position')

        # if previous_position_if is None:
        #     on_land = np.where(self.environment.land_binary_mask == 1)[0]
        # else:
        #     on_land = np.where((self.environment.land_binary_mask == 1) |
        #                        (previous_position_if == 1))[0]
        if len(on_land) == 0:
            logger.debug('No elements hit coastline.')
        else:                
            if self.get_config('biology:min_settlement_age_seconds') == 0.0 :
                # No minimum age input, set back to previous position (same as in interact_with_coastline() from basemodel.py)
                logger.debug('%s elements hit coastline, '
                          'moving back to water' % len(on_land))
                on_land_ID = self.elements.ID[on_land]
                self.elements.lon[on_land] = \
                    np.copy(self.previous_lon[on_land_ID - 1])
                self.elements.lat[on_land] = \
                    np.copy(self.previous_lat[on_land_ID - 1])
                self.environment.land_binary_mask[on_land] = 0
            else:
                #################################
                # Minimum age before settling was input; check age of particle versus min_settlement_age_seconds
                # and strand or recirculate accordingly
                on_land_and_younger = np.where((self.environment.land_binary_mask == 1) & (self.elements.age_seconds < self.get_config('biology:min_settlement_age_seconds')))[0]
                on_land_and_older = np.where((self.environment.land_binary_mask == 1) & (self.elements.age_seconds >= self.get_config('biology:min_settlement_age_seconds')))[0]

                # this step replicates what is done is original code, but accounting for particle age. It seems necessary 
                # to have an array of ID, rather than directly indexing using the "np.where-type" index (in dint64)
                on_land_and_younger_ID = self.elements.ID[on_land_and_younger] 
                on_land_and_older_ID = self.elements.ID[on_land_and_older]

                logger.debug('%s elements hit coastline' % len(on_land))
                logger.debug('moving %s elements younger than min_settlement_age_seconds back to previous water position' % len(on_land_and_younger))
                logger.debug('%s elements older than min_settlement_age_seconds remain stranded on coast' % len(on_land_and_younger))
                
                # refloat elements younger than min_settlement_age back to previous position(s)
                if len(on_land_and_younger) > 0 :
                    # self.elements.lon[np.where(on_land_and_younger)] = np.copy(self.previous_lon[np.where(on_land_and_younger)])  
                    # self.elements.lat[np.where(on_land_and_younger)] = np.copy(self.previous_lat[np.where(on_land_and_younger)])
                    # self.environment.land_binary_mask[on_land_and_younger] = 0 

                    self.elements.lon[on_land_and_younger] = np.copy(self.previous_lon[on_land_and_younger_ID - 1])
                    self.elements.lat[on_land_and_younger] = np.copy(self.previous_lat[on_land_and_younger_ID - 1])
                    self.environment.land_binary_mask[on_land_and_younger] = 0

                # deactivate elements older than min_settlement_age & save position
                # ** function expects an array of size consistent with self.elements.lon
                self.deactivate_elements((self.environment.land_binary_mask == 1) & \
                                         (self.elements.age_seconds >= self.get_config('biology:min_settlement_age_seconds')),
                                         reason='settled_on_coast')        
        
        
    def update(self):
        """Update positions and properties of planktonn particles."""

        # move particles with ambient current
        self.advect_ocean_current()

        # Disabled for now
        if False:
            # Advect particles due to surface wind drag,
            # according to element property wind_drift_factor
            self.advect_wind()
            # Stokes drift
            self.stokes_drift()

        # Turbulent Mixing or settling-only 
        if self.get_config('drift:vertical_mixing') is True:
            self.update_terminal_velocity()  #compute vertical velocities, two cases possible - constant, or same as pelagic egg
            self.vertical_mixing()
        else:  # Buoyancy
            self.update_terminal_velocity()
            self.vertical_buoyancy()

        # Vertical advection
        self.vertical_advection()
        
        # Plankton specific
        if True:
            self.plankton_development()
        
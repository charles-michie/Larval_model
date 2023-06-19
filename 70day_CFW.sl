from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_ROMS_native
from opendrift.readers import reader_ROMS_native_MOANA
from datetime import datetime,timedelta
from opendrift.models.mscmodelv4 import MScLarvaeDrift
import numpy as np
import sys

import argparse
parser = argparse.ArgumentParser(description='Run opendrift simulation')
parser.add_argument('-ym','--yearmonth', type=int, required=True, help='Month and year to seed (yyyymm)')

args = parser.parse_args()

###############################
#Create Readers
###############################
yyyymm = str(args.yearmonth)

mm = int(yyyymm[-2:])
yyyy = int(yyyymm[0:4])

date0 = datetime(yyyy, mm, 1)
mm0 = date0.strftime('%m')
yyyy0 = date0.strftime('%Y')

date1 = date0 + timedelta(days = 31)
mm1 = date1.strftime('%m')
yyyy1 = date1.strftime('%Y')

date2 = date1 + timedelta(days = 31)
mm2 = date2.strftime('%m')
yyyy2 = date2.strftime('%Y')

date3 = date2 + timedelta(days = 31)
mm3 = date3.strftime('%m')
yyyy3 = date3.strftime('%Y')


path0 = 'http://thredds.moanaproject.org:8080/thredds/dodsC/moana/ocean/NZB/v1.9/monthly_avg/nz5km_avg_' + yyyy0 + mm0 + '.nc'
path1 = 'http://thredds.moanaproject.org:8080/thredds/dodsC/moana/ocean/NZB/v1.9/monthly_avg/nz5km_avg_' + yyyy1 + mm1 + '.nc'
path2 = 'http://thredds.moanaproject.org:8080/thredds/dodsC/moana/ocean/NZB/v1.9/monthly_avg/nz5km_avg_' + yyyy2 + mm2 + '.nc'
path3 = 'http://thredds.moanaproject.org:8080/thredds/dodsC/moana/ocean/NZB/v1.9/monthly_avg/nz5km_avg_' + yyyy3 + mm3 + '.nc'


reader0 = reader_ROMS_native_MOANA.Reader(path0)
reader1 = reader_ROMS_native_MOANA.Reader(path1)
reader2 = reader_ROMS_native_MOANA.Reader(path2)
reader3 = reader_ROMS_native_MOANA.Reader(path3)

reader0.multiprocessing_fail = True 
reader1.multiprocessing_fail = True 
reader2.multiprocessing_fail = True 
reader3.multiprocessing_fail = True 

##############################
#Create Simulation Object
###############################

o = MScLarvaeDrift(loglevel=20)
o.add_reader([reader0, reader1, reader2, reader3])
Kxy = 0.1176 # m2/s-1
Kz = 0.01 # m2/s-1  

o.set_config('general:use_auto_landmask', False)
o.set_config('environment:fallback:land_binary_mask', 0)

print(reader0)
print(reader1)
print(reader2)
print(reader3)

o.set_config('environment:fallback:ocean_vertical_diffusivity', Kz) # specify constant ocean_vertical_diffusivity in m2.s-1

o.set_config('drift:vertical_mixing', True)

o.set_config('vertical_mixing:diffusivitymodel', 'constant') # constant >> use fall back values (can be environment (i.e. from reader), windspeed_Large1994 ,windspeed_Sundby1983)

o.set_config('vertical_mixing:timestep', 900.) # seconds - # Vertical mixing requires fast time step  (but for constant diffusivity, use same as model step)

o.set_config('drift:horizontal_diffusivity',Kxy) # using new config rather than current uncertainty

o.set_config('drift:advection_scheme', 'runge-kutta4')

z = -np.random.rand(28000)*100

o.seed_elements( lon = 171.4632, lat = -41.7469, 
                 number=28000, z = z ,
                 radius = 2000,
                 terminal_velocity = -0.001, # setting a large settling velocity on purpose to make seafloor contacts
                 time = [reader0.start_time, reader0.start_time+timedelta(days=28)],
                 origin_marker=10) #CFW
                 


                
o.set_config('biology:min_settlement_age_seconds', 34*86400)
o.set_config('biology:max_settlement_depth', -40)
o.set_config('biology:mortality_daily_rate', 0.15)
o.set_config('drift:max_age_seconds', 70*86400)
o.set_config('biology:heavy_terminal_velocity', -0.0025)
print(o)

start_lon = o.elements_scheduled.lon
start_lat = o.elements_scheduled.lat

o.run(end_time = reader3.end_time,
      time_step=timedelta(seconds = 900),
      time_step_output=timedelta(seconds = 3600 * 3),
      outfile = f'CFW_70day_{yyyy0}{mm0}.nc', export_variables=['lon','lat','status','ID','z','terminal_velocity','origin_marker', 'ocean_vertical_diffusivity','survival', 'trajectory','age_seconds'])
      
print("Run finished")
print(o)

index_of_first, index_of_last = o.index_of_activation_and_deactivation()
long = o.get_property('lon')[0]
lati = o.get_property('lat')[0]
status = o.get_property('status')[0]
origin = o.get_property('origin_marker')[0]
origin = origin[index_of_first, range(long.shape[1])]
lon_end = long[index_of_last, range(long.shape[1])]
lat_end = lati[index_of_last, range(lati.shape[1])]
status_end = status[index_of_last, range(long.shape[1])]
print(o.status_categories)
all = np.column_stack([start_lon, start_lat, lon_end, lat_end, status_end, origin])
np.savetxt(f'CFW_70day_{yyyy0}{mm0}.csv', all, fmt='%.4f', delimiter =',')

sys.stdout = open("Status_CFW.txt", "a")
print(yyyy0)
print(o.status_categories)
sys.stdout.close()
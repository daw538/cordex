# Code by Daniel Williams (Jul 2020)

import numpy as np
import netCDF4 as nc4
import glob
import os
import tqdm
import datetime as dtm

#%% Open CORDEX-SA Datasets

os.chdir('/home/daniel/Documents/meteorology/m6')
input_files = glob.glob('cordex/*.{}'.format('nc'))

# Assign keys for automation of process
CORDEX_SA_vals = [nc4.Dataset(f) for f in sorted(input_files)]
CORDEX_SA_keys = ['81_85','86_90','91_95','96_00','01_05','06_10']
CORDEX_SA_file = dict(zip(CORDEX_SA_keys, CORDEX_SA_vals))

# Read netcdf file to get key coordinates
CORDEX_SA_lat = CORDEX_SA_file['81_85'].variables['lat'][:]
CORDEX_SA_lon = CORDEX_SA_file['81_85'].variables['lon'][:]
CORDEX_time = [(CORDEX_SA_file[i].variables['time'][:]) for i in CORDEX_SA_keys]
CORDEX_time = dict(zip(CORDEX_SA_keys, CORDEX_time))

t0 = dtm.datetime.strptime('01/12/1949', '%d/%m/%Y')

# Check precipitation readings for each cell: assign non-physical values to 
# np.nan and multiply by 86000 to convert units in CORDEX dataset to mm/day
CORDEX_SA_prec, CORDEX_dts = [], []
for idx, f in enumerate(CORDEX_SA_keys):
    dt = [(t0 + dtm.timedelta(days=int(np.floor(i)), 
                              hours=12)) for i in CORDEX_time[f]]
    prec = CORDEX_SA_file[f].variables['pr'][:] * 86400
    prec[prec > 1000000] = np.nan
    CORDEX_SA_prec.append(prec)
    CORDEX_dts.append(dt)
    
CORDEX_SA_all = dict(zip(CORDEX_SA_keys, CORDEX_SA_prec))
CORDEX_dts = dict(zip(CORDEX_SA_keys, CORDEX_dts))

for f in CORDEX_SA_vals:
    f.close()

#%% Define coarse analysis grid
india_lat = np.arange(6,39,1)
india_lon = np.arange(67,100,1)
offset = 0.5
dlat = 0
dlon = 1

#%% Find CORDEX_SA cell midpoints

# The following function calculates the mid-points of every cell within a
# rotated co-ordinate grid. Note this code has not been tested where the
# rotated grid crosses the poles or the antimeridian (180E/W)
def grid_midpoints(grid_lat,grid_lon):
    cdx_midpts = []
    for i in range(grid_lat.shape[0]-1):
        cdx_midpts_row = []
        for j in range(grid_lat.shape[1]-1):
            # Indentify co-ordinates of cell vertices, using m x n notation
            y11 = grid_lat[i][j]
            y12 = grid_lat[i][j+1]
            y21 = grid_lat[i+1][j]
            y22 = grid_lat[i+1][j+1]
            x11 = grid_lon[i][j]
            x12 = grid_lon[i][j+1]
            x21 = grid_lon[i+1][j]
            x22 = grid_lon[i+1][j+1]
            # Aggregate points into an array for whole cell
            cdx_cell = np.array([[[y11,x11],[y12,x12]],
                                 [[y21,x21],[y22,x22]]])
            # Convert angles to radians then convert spherical to cartesian
            # co-ordinates. Perform vector averaging then convert back into
            # spherical co-ordinate system to find true cell mid-point
            cdx_cell_r = (cdx_cell*np.pi)/180.0
            x = np.cos(cdx_cell_r[:,:,dlat]) * np.cos(cdx_cell_r[:,:,dlon])
            y = np.cos(cdx_cell_r[:,:,dlat]) * np.sin(cdx_cell_r[:,:,dlon])
            z = np.sin(cdx_cell_r[:,:,dlat])
            x_avg = np.mean(x)
            y_avg = np.mean(y)
            z_avg = np.mean(z)
            
            lon2 = (np.arctan2(y_avg, x_avg)*180)/np.pi
            lat2 = (np.arcsin(z_avg)*180)/np.pi
            mp = [lat2,lon2]
            cdx_midpts_row.append(mp)
        cdx_midpts.append(cdx_midpts_row)
        
    cdx_midpts = (np.asarray(cdx_midpts))
    return cdx_midpts



def whichcell(midpoints, lats, lons):
    # The following loops iterate through the cells defined as our big grid
    # (a 32x32deg area with 1 deg resolution) to find which mid-points of the 
    # CORDEX-SA dataset reside within each cell
    ingrid_idx = []
    for i in range(len(lats)-1):
        incells_idx = []
        for j in range(len(lons)-1):
            lim_l = lons[j]
            lim_r = lons[j+1]
            lim_u = lats[i+1]
            lim_d = lats[i]
            # Creates list of indices of midpoints that fall within defined cell
            incell_idx = list(zip(*np.where(((midpoints[:,:,dlat] >= lim_d) & 
                                             (midpoints[:,:,dlat] <= lim_u)) & 
                                            ((midpoints[:,:,dlon] >= lim_l) & 
                                             (midpoints[:,:,dlon] <= lim_r)))))
            incells_idx.append(np.asarray(incell_idx))
        ingrid_idx.append((incells_idx))
    
    return ingrid_idx



def india_rain_day(file,t):
    ingrid_flat = np.reshape(ingrid_idx, (len(ingrid_idx)**2,1))
    avg_prec = []
    for cdx_pts in ingrid_flat:
        cell_avg_prec = [CORDEX_SA_all[file][t,pt[dlat],pt[dlon]]
                            for pt in cdx_pts[0]]
        avg_prec.append(np.mean(cell_avg_prec))
            
    grid_avg_prec = np.reshape(avg_prec, (len(ingrid_idx),len(ingrid_idx)))
    return grid_avg_prec


# Function to create new netcdf file with desired information
def save_netcdf(file):
    try: ncfile.close()
    except: pass
    if int(file[0]) > 3: fname = '19' + file
    else: fname = '20' + file
    ncfile = nc4.Dataset('cordex_ind/cordex_india_' + fname + '.nc',
                         mode='w', format='NETCDF4_CLASSIC')
    ncfile.createDimension('lat', 32)
    ncfile.createDimension('lon', 32)
    ncfile.createDimension('time', None)
    ncfile.title='CORDEX-SA ' + fname + ' India'
    
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'days since 1949-12-01 00:00:00'
    time.long_name = 'time'
    pr = ncfile.createVariable('pr',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
    pr.units = 'mm/day' # millimetres per day
    pr.standard_name = 'precipitation'
    
    lat[:] = india_lat[:-1] + offset # south pole to north pole
    lon[:] = india_lon[:-1] + offset
    pr[:,:,:] = india_rain_arr # Appends data along unlimited dimension
    time[:] = CORDEX_time[file]
    #print(ncfile)
    print('Successfully created .nc file for the period' + fname)
    ncfile.close()
#%%
cdx_midpts = grid_midpoints(CORDEX_SA_lat,CORDEX_SA_lon)

ingrid_idx = whichcell(cdx_midpts, india_lat, india_lon)
ingrid_idx = np.asarray(ingrid_idx)  

#%%

# Do the heavy lifting and create the new file
for file_key in CORDEX_SA_keys:
    india_rain = [india_rain_day(file_key,t) for t in 
                  tqdm.tqdm(range(len(CORDEX_dts[file_key])))]
    india_rain_arr = np.asarray(india_rain)    
    
    save_netcdf(file_key)






















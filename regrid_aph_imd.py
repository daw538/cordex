# Code by Daniel Williams (Jul 2020)

import numpy as np
import netCDF4 as nc4
import glob
import os
import tqdm
import datetime as dtm
import warnings

#from matplotlib import rc
#rc("font", family="sans", size=8)

#%% Open datasets
os.chdir('/home/daniel/Documents/meteorology/m6')
directory = input('Enter name of directory containing .nc files you wish to regrid: \n')
input_files = sorted(glob.glob(directory + '/*.{}'.format('nc')))
print(f'{len(input_files)} files were found in this directory.')
#%% Find midpoints each cell within the dataset

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


# Within the large grid we can find which datapoints it contains
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



# For each location within the large gridset, calculate the cell-averaged
# precipitation reading for each day recorded.
def india_rain_day(file,t):
    ingrid_flat = np.reshape(ingrid_idx, (len(ingrid_idx)**2,1))
    avg_prec = []
    for cdx_pts in ingrid_flat:
        cell_avg_prec = [file[t,pt[dlat],pt[dlon]]
                            for pt in cdx_pts[0]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            avg_prec.append(np.nanmean(cell_avg_prec))
            
    grid_avg_prec = np.reshape(avg_prec, (len(ingrid_idx),len(ingrid_idx)))
    return grid_avg_prec


# Export the regridded data as a new netCDF file
def save_netcdf(fname):
    try: ncfile.close()
    except: pass
    #if int(file[0]) > 3: fname = '19' + file
    #else: fname = '20' + file
    ncfile = nc4.Dataset(directory + '_ind/imd_india_' + fname + '.nc',
                         mode='w', format='NETCDF4_CLASSIC')
    ncfile.createDimension('lat', 32)
    ncfile.createDimension('lon', 32)
    ncfile.createDimension('time', None)
    ncfile.title='APHRODITE ' + fname + ' India'
    
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
    time[:] = dts
    #print(ncfile)
    print(' ...' + fname + '.nc created')
    ncfile.close()

#%%
india_lat = np.arange(6,39,1)
india_lon = np.arange(67,100,1)
offset = 0.5
dlat = 0
dlon = 1

base = nc4.Dataset(input_files[0])
DATA_lat = base.variables['IMD_lat'][:]
DATA_lon = base.variables['IMD_lon'][:]
DATA_grid = np.meshgrid(DATA_lon, DATA_lat)

DATA_midpts = grid_midpoints(DATA_grid[1],DATA_grid[0])
ingrid_idx = whichcell(DATA_midpts, india_lat, india_lon)
ingrid_idx = np.asarray(ingrid_idx, dtype=object)
base.close()

#%%

for f in tqdm.tqdm(range(1)):#range(len(input_files))):
    data = nc4.Dataset(input_files[f])
    
    year = input_files[f][-7:-3]
    t0_cdx = dtm.datetime.strptime('01/12/1949', '%d/%m/%Y')
    t0_data = dtm.datetime.strptime('01/01/' + str(year), '%d/%m/%Y')
    
    DATA_prec = data.variables['IMD_rain'][:]
    DATA_prec[DATA_prec < 0] = np.nan

    try:
        DATA_time = data.variables['time'][:]
        dates = [(t0_data + dtm.timedelta(minutes=i)) for i in DATA_time]
    except KeyError:
        print(' File contains no datetime values. ' + 
              'Values will be interpolated from filename.')
        DATA_prec = np.rollaxis(DATA_prec, 2)
        DATA_time = np.arange(DATA_prec.shape[0])
        t0_data = dtm.datetime.strptime('01/01/' + str(year), '%d/%m/%Y')
        dates = [(t0_data + dtm.timedelta(days=int(i))) for i in DATA_time]
    finally:
        dts = [(i - t0_cdx).days for i in dates]        
        
        india_rain = [india_rain_day(DATA_prec,t) for t in 
                      range(len(dts))]
        india_rain_arr = np.asarray(india_rain)    
        
        #save_netcdf(year)
        data.close()



#%%

import pandas as pd





locs = pd.read_csv('/home/daniel/Documents/meteorology/m6/indian_locations.csv')


#locs = locs.sort_values('lat')
locs



#%%

cl = y[:,0][locs['ilat'].values]
ck = x[0,:][locs['ilon'].values]






#%% Testing plot
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap#, addcyclic, shiftgrid

#import matplotlib.animation as anim
colours = itertools.cycle(('C2','k','C1',))
# Define basemap extents
llon = 60
ulon = 110
llat = 0
ulat = 45
m = Basemap(llcrnrlon = llon, llcrnrlat = llat,
            urcrnrlon = ulon, urcrnrlat = ulat,
            resolution='l', projection='cyl')

fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')

x, y = np.meshgrid(np.arange(67.5,99,1), np.arange(6.5,38,1))
#x, y = np.meshgrid(india_lon, india_lat)
m.contourf(x, y, india_rain_arr[0,:,:], cmap='Blues')
#for i in range(len(locs)):
#    m.scatter(locs['glon'][i], locs['glat'][i])
m.scatter(ck,cl, marker='^', color='C3')

m.drawmeridians(np.arange(67, 100, 8), linewidth=0.5, dashes=[4, 2],
                labels=[0,0,0,1], color='dimgrey')
m.drawparallels(np.arange(6, 39, 8), linewidth=0.5, dashes=[4, 2],
                labels=[1,0,0,1], color='dimgrey')
m.drawcountries(color='k', linewidth=1, zorder=5)
m.drawcoastlines(linewidth=1, zorder=5)
m.drawmapboundary()

plt.xlim([llon,ulon])
plt.ylim([llat,ulat])
#plt.savefig('cordex_inbiggrid.png', bbox_inches='tight', dpi=200)
plt.show()


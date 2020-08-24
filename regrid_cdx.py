# Code by Daniel Williams (Jul 2020)

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from mpl_toolkits.basemap import Basemap#, addcyclic, shiftgrid
import glob
import os
import tqdm
import datetime as dtm

#from matplotlib import rc
#rc("font", family="sans", size=8)

#%% Open CORDEX-SA Datasets

os.chdir('/home/daniel/Documents/meteorology/m6')
input_files = glob.glob('cordex/*.{}'.format('nc'))

CORDEX_SA_vals = [nc4.Dataset(f) for f in sorted(input_files)]
CORDEX_SA_keys = ['81_85','86_90','91_95','96_00','01_05','06_10']
CORDEX_SA_file = dict(zip(CORDEX_SA_keys, CORDEX_SA_vals))

CORDEX_SA_lat = CORDEX_SA_file['81_85'].variables['lat'][:]
CORDEX_SA_lon = CORDEX_SA_file['81_85'].variables['lon'][:]
CORDEX_time = [(CORDEX_SA_file[i].variables['time'][:]) for i in CORDEX_SA_keys]
CORDEX_time = dict(zip(CORDEX_SA_keys, CORDEX_time))

t0 = dtm.datetime.strptime('01/12/1949', '%d/%m/%Y')
#dt = [(t0 + dtm.timedelta(days=int(np.floor(i)), hours=12)) for i in CORDEX_time[0]]

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

#cdx boundaries = [[-15.232, 45.25],[19.864, 115.53]]
#imd boundaries = [[6.5,38.5],[66.5,100]]
#phd boundaries = [[-13.8750,54.8750],[60.1250,149.8750]]
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
#np.savetxt('cdxmpts.csv', cdx_midpts, delimiter=',')

# Accessing binary array for a big grid cell
#print(ingrid[m,n,:,:])

ingrid_idx = whichcell(cdx_midpts, india_lat, india_lon)
ingrid_idx = np.asarray(ingrid_idx)  

#%%

#file_key = '06_10'

for file_key in CORDEX_SA_keys:
    india_rain = [india_rain_day(file_key,t) for t in 
                  tqdm.tqdm(range(len(CORDEX_dts[file_key])))]
    india_rain_arr = np.asarray(india_rain)    
    
    save_netcdf(file_key)




#%%

#%% Testing plot
import itertools
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
#print(len(np.arange(67.5,99,1)))
#x, y = m(np.arange(67.5,99,1), np.arange(6.5,38,1))
x, y = np.meshgrid(np.arange(67.5,99,1), np.arange(6.5,38,1))
#x, y = np.meshgrid(india_lon, india_lat)
m.contourf(x, y, india_rain_arr[0,:,:], cmap='Blues')

#plt.scatter(cdx_midpts[:,:,1],cdx_midpts[:,:,0],
#               c=gg, cmap='Blues', s=1, vmin=-1, vmax=2)
'''
for i in ingrid_idx:
    for j in i:
        colour = next(colours)
        ax.scatter(cdx_midpts[j[:,0],j[:,1],dlon],
                   cdx_midpts[j[:,0],j[:,1],dlat],
                   s=4, color=colour, alpha=0.5)

for i in range(len(india_lat)):
    ax.axhline(india_lat[i], color='lightgrey', linewidth=1)
    ax.axvline(india_lon[i], color='lightgrey', linewidth=1)
'''
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


#%% Testing plot
import matplotlib.animation as anim
import datetime as dtm
# Define basemap extents
llon = 67
ulon = 99
llat = 6
ulat = 38
m = Basemap(llcrnrlon = llon, llcrnrlat = llat,
            urcrnrlon = ulon, urcrnrlat = ulat,
            resolution='l', projection='cyl')

fig, ax = plt.subplots(figsize=(5,5))
ax.set_aspect('equal')

x, y = np.meshgrid(np.arange(67.5,99,1), np.arange(6.5,38,1))

cvals = np.arange(1,101,10)
cont = m.contourf(x, y, india_rain_arr[11,:,:], levels=10, vmin=2, vmax=60, cmap='Blues')

m.drawmeridians(np.arange(67, 100, 8), linewidth=0.5, dashes=[4, 2],
                labels=[0,0,0,1], color='dimgrey')
m.drawparallels(np.arange(6, 39, 8), linewidth=0.5, dashes=[4, 2],
                labels=[1,0,0,1], color='dimgrey')
m.drawcountries(color='k', linewidth=1, zorder=5)
m.drawcoastlines(linewidth=1, zorder=5)
m.drawmapboundary()

#plt.xlim([llon,ulon])
#plt.ylim([llat,ulat])
#plt.savefig('cordex_inbiggrid.png', bbox_inches='tight', dpi=200)

dt_start = "01/12/1949"
t0 = dtm.datetime.strptime(dt_start, "%m/%d/%Y")

def animate(i):
    if i<12:
        global cont
        z = india_rain_arr[i,:,:]
        for c in cont.collections:
            c.remove()
        cont = m.contourf(x,y,z, levels=10, vmin=2, vmax=60, cmap='Blues')
        dt = (t0 + dtm.timedelta(days=int(np.floor(CORDEX_time[0][i])), hours=12))
        plt.title(f'{dt.strftime("%d/%m/%Y")}')
        #print(i, dt)
        return cont



movie = anim.FuncAnimation(fig, animate, frames=len(india_rain_arr), repeat=False)
#movie.save('animation.mp4', writer=anim.FFMpegWriter())
#plt.show()






















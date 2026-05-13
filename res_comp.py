import h5py
import numpy as np    
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import matplotlib as mpl
import sys
mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

plt.style.use("seaborn-paper")
color_map = plt.get_cmap('tab10')

simulation_directory = str(sys.argv[1]) 
output_directory = str(sys.argv[2]) 

#### FUNCTIONS ####
def calculate_cell_size(volume):
    return 2 * np.cbrt(volume * 3 /(4 * np.pi))

def read_csv_file(csv_file):
    data_file = pd.read_csv(csv_file).to_numpy()
    return data_file[:,0], data_file[:,1] 

### PARAMETER CONSTANTS ###
filename = simulation_directory + "./snap_000.hdf5" 
with h5py.File(filename,'r') as f:
    parameters = dict(f['Parameters'].attrs)
    cells_per_dim = int(np.cbrt(len(f['PartType0']['Density'][()])))
boxsize = parameters["BoxSize"] # boxsize in kpc
center_boxsize = 10 # the boxsize of the inner box

#### Idealized simulations compared #######
cgols_resolution = 5 # pc - 2018 - 2024/5
smuag_resolution = 2 # https://arxiv.org/pdf/2006.16315 - r2/r4 - most similar to ours - 2020

##### Cosmological simulations that are being compared ####
tng_dist,tng_res = read_csv_file('tables/tng_res_m135.csv') # https://arxiv.org/pdf/2005.09654 - # Cosmological eyballing 150 kpc + using the AREPO equation for static nfw  gives us a log10(150/rvir)= -0.52? This gives us a virial radius of ~ 500 kpc?
tng_dis_resc = 500*pow(10, tng_dist) # log10(r/500 kpc)= x = 10^(x)*500kpc

# For the architect paper, Figure 11 is spkit between the ISK, CGM, and IGM with thre grey lines being placed at 0.1 R200 and R200
architect26_dist, architect26_res = read_csv_file('tables/ArchME_2026.csv') # https://arxiv.org/pdf/2602.13392 - used ME (mechanical)
## Both gibble and suresh were digitized from  https://arxiv.org/pdf/2602.13392
gible24_dist, gible24_res = read_csv_file('tables/gible_2024.csv') # https://arxiv.org/pdf/2307.11143 -> Cosmological - rahul's paper, but taken from https://arxiv.org/pdf/2602.13392
suresh19_dist, suresh19_res = read_csv_file('tables/suresh_19.csv') # cosmological - Zooming in on accretion – II. Cold circumgalactic gas simulated with a super-Lagrangian refinement schem, but taken from https://arxiv.org/pdf/2602.13392

######### SIMULATION DATA #########
snaps = [150, 175, 200] # Pick three snapshots at steady state
coloring = color_map(np.linspace(0, 1, 9))
j = [2, 1, 0.5] # changing the cell size by 8 doubles the cell size by design - corresponds to 150^3, 300^3, 600^3
labeling = [r"$150^3$",  r"$300^3$", r"$600^3$"] # resolution

fig = plt.figure(figsize=(5.1,4))
fig.set_rasterized(True)
ax1 = fig.add_subplot(111)
for i, snap in enumerate(snaps):
    data = {}
    print("Reading from snaphot_%0.03d.hdf5" % snap)
    filename = simulation_directory + "./snap_%03d.hdf5" % snap
    with h5py.File(filename,'r') as f:
        for key in ['Coordinates', 'Density', 'Masses']:
            data[key] = f['PartType0'][key][()]
        header = dict(f['Header'].attrs)
        parameters = dict(f['Parameters'].attrs)
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    masses = data["Masses"] 
    volume = masses/density 

    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 

    cell_size = calculate_cell_size(volume) 
    cs_log10 = np.log10(cell_size*j[i])

    cell_profile, c_bin_edge, _ = stats.binned_statistic(radius, cs_log10, bins=200, statistic='median', range=(0, 30))
    ax1.plot(c_bin_edge[:-1], cell_profile, label=labeling[i], lw=2.0, color=coloring[i])

    cf_min, c_bin_edge, _ = stats.binned_statistic(radius, cs_log10, bins=200, statistic='min', range=(0, 30))
    cf_max, c_bin_edge, _ = stats.binned_statistic(radius, cs_log10, bins=200, statistic='max', range=(0, 30))

    ax1.fill_between(c_bin_edge[:-1], cf_min, cf_max, alpha=0.10) 

ax1.plot(tng_dis_resc, tng_res, linestyle='dashed', label="TNG50-1", color=coloring[3]) 
ax1.plot(architect26_dist*100, np.log10(architect26_res), linestyle='dashed', label="ARCHITECTS", color=coloring[4]) 
ax1.plot(gible24_dist*100, np.log10(gible24_res), linestyle='dashed', label="GIBLE", color=coloring[5])
ax1.plot(suresh19_dist*100, np.log10(suresh19_res), linestyle='dashed', label="Suresh+19", color=coloring[6])

ax1.axhline(np.log10(cgols_resolution/1000), linestyle='dotted', label="CGOLS", color=coloring[7]) 
ax1.axhline(np.log10(smuag_resolution/1000), linestyle='dotted', label="SMAUG - R2/R4", color=coloring[8]) 

ax1.set(xlim=(0.5, 20), ylim=(-3, 0.75))
ax1.set_xlabel(r"Radius [kpc]",  fontsize=11)
ax1.set_ylabel(r"Cell Size [$ \rm log_{10}(kpc)$]", fontsize=11)

ax1.legend(loc="upper left", ncol=3, frameon=False, fontsize=9.5) # ,  fontsize=10.5, ncol=2)

ax1.tick_params(axis='both', which='major', labelsize=9)

plt.savefig(output_directory + "mvhist.pdf", bbox_inches='tight', dpi=200) 
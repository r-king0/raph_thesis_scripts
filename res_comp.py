


import numpy as np    
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from utilities import load_snapshots # taken from utilities.py
import matplotlib as mpl
import sys

mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix
plt.style.use("seaborn-paper")

simulation_directory = str(sys.argv[1]) 
output_directory = str(sys.argv[2]) 

#### FUNCTIONS ####

def calculate_cell_size(volume):
    return 2 * np.cbrt(volume * 3 /(4 * np.pi))

def read_csv_file(csv_file):
    data_file = pd.read_csv(csv_file).to_numpy()
    return data_file[:,0], data_file[:,1] 

boxsize = 100 # boxsize in kpc
center_boxsize = 10 # the boxsize of the inner box

# Idealized simulations that are being compared
cgols_resolution = 5 # pc - 2018 - 2024/5
smuag_resolution = 2 # https://arxiv.org/pdf/2006.16315 - r2/r4 - most similar to ours - 2020

# Cosmological simulations that are being compared
tng_dist,tng_res = read_csv_file('tables/tng_res_m135.csv') # https://arxiv.org/pdf/2005.09654 - # Cosmological eyballing 150 kpc + using the AREPO equation for static nfw  gives us a log10(150/rvir)= -0.52? This gives us a virial radius of ~ 500 kpc?
tng_dis_resc = 500*pow(10, tng_dist) # log10(r/500 kpc)= x = 10^(x)*500kpc

# 0.1 R200 is ~10 kpc and 1 R200 ~100 kpc. Multiple everything by 100 to get "actual" distance
gible24_dist, gible24_res = read_csv_file('tables/gible_2024.csv') # https://arxiv.org/pdf/2307.11143 -> Cosmological - rahul's paper, but taken from https://arxiv.org/pdf/2602.13392

suresh19_dist, suresh19_res = read_csv_file('tables/suresh_19.csv') # cosmological - Zooming in on accretion – II. Cold circumgalactic gas simulated with a super-Lagrangian refinement schem, but taken from https://arxiv.org/pdf/2602.13392

# For the architect paper, Figure 11 is spkit between the ISK, CGM, and IGM with thre grey lines being placed at 0.1 R200 and R200
architect26_dist, architect26_res = read_csv_file('tables/ArchME_2026.csv') # https://arxiv.org/pdf/2602.13392 - used ME (mechanical)

######### DISK PARAMETERS #########
z_stars = 0.15 # the scale height of the gas disk -> based on the mass at R_stars. Note that in the plots, we are plotting with respect to radius and not the radial coordinate, which Rs is in.
R_stars = 0.8  # Stellar scale radius
disk_radial = R_stars*3 # the scale length of the gas disk

######### SIMULATION DATA #########
snaps = [150, 175, 200] # pick which ever snapshots you want to iterate through. I always go with the first, middle, and last snapshots
color_map = plt.get_cmap('tab10')
coloring = color_map(np.linspace(0, 1, 9))
j = [2, 1, 0.5] # 150, 1 (actual), 0.5 (600)
fig = plt.figure(figsize=(5.1,4))
fig.set_rasterized(True)
ax1 = fig.add_subplot(111)
labeling = [r"$150^3$" , r"$300^3$", r"$600^3$"]
keys = ['Coordinates', 'Density', 'Masses']
for i, snap in enumerate(snaps):
    print("Reading from snaphot_%0.03d.hdf5" % snap)
    data, header, parameters = load_snapshots(simulation_directory + "/snap_%03d.hdf5" % snap, keys)
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    masses = data["Masses"] 
    volume = masses/density 

    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 
    radial_coord = np.sqrt(rad_x**2 + rad_y**2)
    rinject = parameters["injection_radius"]
    cell_size = calculate_cell_size(volume) 
    cs_log10 = np.log10(cell_size*j[i])

    # 1.5 and 1.2 are fudge factor as the disk has expanded a bit throughout relaxation. The tuple is included for future versions of the code. 
    # the disk is a region between np.abs(z_stars) in z and inside a pancake of radial coordinate 3 *disk_radial.
    disk_cells = tuple([(abs(rad_z) <= z_stars*1.5) & (radial_coord <= disk_radial*1.2)]) 
    outside_disk = ~disk_cells[0] # [0] is becuaose of the tuple

    print("Making Profiles")
    cell_profile_ang, c_bin_edge, _ = stats.binned_statistic(radius[outside_disk], cs_log10[outside_disk], bins=600, statistic='median', range=(0, 30))
    ax1.plot(c_bin_edge[:-1], cell_profile_ang, label=labeling[i], lw=2.0, color=coloring[i])

    cs_indices = np.argsort(cell_size)
    
    cf_min, c_bin_edge, _ = stats.binned_statistic(radius[outside_disk], cs_log10[outside_disk], bins=200, statistic='min', range=(0, 30))
    cf_max, c_bin_edge, _ = stats.binned_statistic(radius[outside_disk], cs_log10[outside_disk], bins=200, statistic='max', range=(0, 30))

    ax1.fill_between(c_bin_edge[:-1], cf_min, cf_max, alpha=0.10) # , edgecolor="black") # , color=coloring[i])

print("Plotting results from other work")
ax1.plot(tng_dis_resc, tng_res, linestyle='dashed', label="TNG50-1", color=coloring[3]) # , color="red" )#  , olor=coloring[5]) 
ax1.plot(architect26_dist*100, np.log10(architect26_res), linestyle='dashed', label="ARCHITECTS", color=coloring[4]) # , color="darkblue" )
ax1.plot(gible24_dist*100, np.log10(gible24_res), linestyle='dashed', label="GIBLE", color=coloring[5])# , color="crimson" )
ax1.plot(suresh19_dist*100, np.log10(suresh19_res), linestyle='dashed', label="Suresh+19", color=coloring[6])

ax1.axhline(np.log10(cgols_resolution/1000), linestyle='dotted', label="CGOLS", color=coloring[7]) # , color="purple") # 5pc/1000pc * 1 kpc = 
ax1.axhline(np.log10(smuag_resolution/1000), linestyle='dotted', label="SMAUG - R2/R4", color=coloring[8]) # , color="teal" ) # , color=coloring[4]) # 5pc/1000pc * 1 kpc = 

ax1.set(xlim=(0.5, 20), ylim=(-3, 0.75))
ax1.set_xlabel(r"Radius [kpc]",  fontsize=11)
ax1.set_ylabel(r"Cell Size [$\rm log_{10}(kpc)$]", fontsize=11)

ax1.legend(loc="upper left", ncol=3, frameon=False, fontsize=9.5) # ,  fontsize=10.5, ncol=2)
ax1.tick_params(axis='both', which='major', labelsize=9)

plt.savefig(output_directory + "mvhist_no_disk.pdf", bbox_inches='tight', dpi=200) 
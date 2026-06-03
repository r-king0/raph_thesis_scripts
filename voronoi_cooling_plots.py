'''
    Generates slices for the cooling rate, metallicity, and electron abundances.
'''
import sys
import time 
import numpy as np  
import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from scipy.spatial import cKDTree
from scipy import stats
mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

### IMPORTANT FUNCTIONS ### 
from utilities import load_snapshots, Temp_S, sol_in, sol_out, mean_molecular_weight, plot_edge, plot_face, plot_edge_cr, plot_face_cr, make_cbar
### PHYSICAL CONSTANTS ###
from utilities import PROTON_MASS_GRAMS, gamma, kb, z_solar
### Units
from utilities import UnitMass_in_g, UnitTime_in_s, UnitEnergy_in_cgs, UnitDensity_in_cgs, UnitPressure_in_cgs
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS

#### Configuration Options ####
FACE_ON = False # Was predominantly used for examnining initial conditions and was rarely used for simulation analysis outside of diagnostics
EXTENDED = False

simulation_directory = str(sys.argv[1]) 
output_directory = str(sys.argv[2]) 

################################
if FACE_ON: print("FACE_ON enabled.")
else: print("FACE_ON disabled. Output will be edge-on")

keys_0 = []
filename = simulation_directory + "/snap_000.hdf5" 
data_0, header_0, parameters_0 = load_snapshots(filename, keys_0)
boxsize = parameters_0["BoxSize"] # boxsize in kpc
inner_boxsize = 10
angle_l = 60
halfbox = boxsize/2
dx = inner_boxsize/300
eps = dx/1e4
halfbox_inner = inner_boxsize/2 
lower_bound, upper_bound = halfbox - dx*6, halfbox + eps*6

if EXTENDED: 
    n_bins = 450
    deviation = 20
    box_range = 100
    prof_bins = 300
    upper_x = 20
else: 
    deviation = 5
    box_range = inner_boxsize
    prof_bins = 150
    upper_x = 5
    n_bins = 300

histb_l = boxsize/2 - deviation # boundary of histogram - lower bound
histb_h = boxsize/2  + deviation # boundary of histogram - upper bound
linethresh = 1e-25

def plot_quant(coord, quantity, ax, prof_bins, stat, ylabel, label, color):
    quantity_stat, r_edge, _ = stats.binned_statistic(coord, quantity, bins=prof_bins, statistic=stat, range=(0, upper_x))
    ax.plot(r_edge[:-1], quantity_stat, label=label, color=color)
    ax.set(xlim=(0, upper_x))
    ax.set_ylabel(ylabel, fontsize=12)
    if FACE_ON: ax.set_xlabel("Radial Distance [kpc]", fontsize=12)
    else: ax.set_xlabel("Radius [kpc]", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

######### SIMULATION DATA #########
start = time.time()

keys = ['Coordinates', 'InternalEnergy', 'PassiveScalars', 'CoolingRate', 'MetallicCoolingRate', 'ElectronAbundance']
for i in np.arange(0,  101, 10): # select the snapshot range to go through
    filename = simulation_directory + "/snap_%03d.hdf5" % i
    data, header, parameters = load_snapshots(filename, keys)
    coord = data["Coordinates"]
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
    t = header["Time"]
    times = t*1000

    metallicities = data["PassiveScalars"]
    cooling_rate = data["CoolingRate"] # For now..newer simulations return the negaive by default. 
    z_cooling_rate = data["MetallicCoolingRate"]
    xe = data["ElectronAbundance"]
    temperature = Temp_S(xe, internal_energy)

    ''' Get the radial distance of the box'''
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2) 

    if FACE_ON: 
        mask = (z_coord >=lower_bound) & (z_coord <= upper_bound) # & (radial_coord <= inner_boxsize/2*np.sqrt(2))
        r_coord = radial_coord[mask] 
        temps = temperature[mask]
    else:
        mask = (y_coord >= lower_bound) & (y_coord <= upper_bound) # & (radius <= inner_boxsize/2*np.sqrt(3))

        if EXTENDED: z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound)
        else: z_mask = (y_coord >= lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound) & (radius <= inner_boxsize/2*np.sqrt(3))

        theta = np.arccos(np.abs(rad_z)/(radius + eps))*180/np.pi 
        angular_region = (np.abs(theta) <= 60) # Excludes anything with absolute angles greater than 60 
        r_coord = radius[z_mask]

    tree = cKDTree(coord[mask]) # coordinates change with each snapshot so keep it different

    r_face = radial_coord[mask] 
    r_z = radius[z_mask]

    ### PLOTS ###
    fig = plt.figure(figsize=(15,9)) # 20/12 = 15/9
    fig.set_rasterized(True)    
    ax1 = fig.add_subplot(2,3,1)

    # specifically for the cooling rate 
    if FACE_ON: plot_face_cr(ax1, coord[mask], cooling_rate[mask], n_bins*2, halfbox, inner_boxsize, -1e-23, 1e-23, histb_l, histb_h, linethresh=linethresh, tree=tree)
    else: plot_edge_cr(ax1, coord[mask], cooling_rate[mask], n_bins*2, halfbox, inner_boxsize, -1e-23, 1e-23, histb_l, histb_h, linethresh=linethresh, tree=tree)
    cr_mesh = ax1.collections[0]
    cr_mesh.set_cmap("cmr.iceburn_r")
    cbar = plt.colorbar(cr_mesh, ax = ax1, label=r'Cooling Rate [$log_{10}(erg \,s^{-1} \, cm^{3})$]')
    labels_cooling = [-1e-23, -1e-24, -1e-25, 0, 1e-25, 1e-24, 1e-23]    
    cbar.set_ticks(labels_cooling)

    background_rect = patches.Rectangle((0, 0.78), width=1, height=0.2, color='black', alpha=0.25, transform=ax1.transAxes, fill=True)
    ax1.add_patch(background_rect)
    ax1.text(0.01, 0.96,"t = %0.3f Myr" % times, transform=ax1.transAxes, color="white", fontsize=12)
    ax1.text(0.01, 0.92,'M82/LMC - Disk Z Test', transform=ax1.transAxes, color="white", fontsize=12)
    ax1.text(0.03, 0.88,r"- $Z_{bg}= 0.00, \alpha = 0.6$", transform=ax1.transAxes, color="white", fontsize=12)
    ax1.text(0.03, 0.84,r"- $Z_{disk}= 0.02, \beta = 0.6$", transform=ax1.transAxes, color="white", fontsize=12)
    ax1.text(0.03, 0.80,r"- $\dot{M}_{SFR} = 10 M_\odot \, yr^{-1}$, $R_{inject}$ = 300 pc", transform=ax1.transAxes, color="white", fontsize=12)

    # 2D VELOCITY CENTER VORONOI SLICE 
    ax2 = fig.add_subplot(2,3,2)

    if FACE_ON: plot_face(ax2, coord[mask], metallicities[mask]/z_solar, n_bins*2, halfbox, inner_boxsize, 0, 4.0,  histb_l, histb_h, log=False, tree=tree)
    else: plot_edge(ax2, coord[mask], metallicities[mask]/z_solar, n_bins*2, halfbox, inner_boxsize, 0.0, 4.0,  histb_l, histb_h, log=False,  tree=tree)
    Z_mesh = ax2.collections[0]
    Z_mesh.set_cmap("cmr.ember")
    cbar2 = make_cbar(Z_mesh, ax2, 0.02, 'Metallicity', [0.00, 1.0, 2.0, 3.0, 4.0], log=False)

    # 2D TEMPERATURE CENTER VORONOI SLICE 
    ax3 = fig.add_subplot(2,3,3)
    if FACE_ON: plot_face(ax3, coord[mask], xe[mask], n_bins*2, halfbox, inner_boxsize, 0, 1,  histb_l, histb_h,  log=False, tree=tree)
    else: plot_edge(ax3, coord[mask], xe[mask], n_bins*2, halfbox, inner_boxsize, 0, 1,  histb_l, histb_h, log=False, tree=tree)
    xe_mesh = ax3.collections[0]
    xe_mesh.set_cmap("cmr.amethyst")
    cbar3 = make_cbar(xe_mesh, ax3, 0.02, r'$\rm n_e/n_H$', [0.0, 0.25, 0.5, 0.75, 1.0 ], log=False)

    ax4= fig.add_subplot(2,3,4)
    if FACE_ON:
        sSc, sSr,  _ = stats.binned_statistic(radial_coord[mask], cooling_rate[mask], statistic='mean', bins=200)
        col_sScnh = np.where(sSc > 0, "blue", "crimson")
        sSc[sSc < 0] *= -1 # the cooling rate should be posible always 
        NC_points = np.array([sSr[:-1], np.log10(sSc)]).T
        crS_nh = np.array([NC_points[:-1], NC_points[1:]]).transpose(1, 0, 2)
        crS_nhlc = LineCollection(crS_nh, colors=col_sScnh[:-1], linewidth=1.5, label=r"$\Lambda_{net}$", linestyle="solid")
        ax4.set_ylim(-29,-20)
        ax4.add_collection(crS_nhlc)
        ax4.set_xlabel("Radial Distance [kpc]", fontsize=12)
    else:
        sSc, sSr,  _ = stats.binned_statistic(radius[mask], cooling_rate[mask], statistic='mean', bins=200)
        
        col_sScnh = np.where(sSc > 0, "blue", "crimson")
        sSc[sSc < 0] *= -1 # the cooling rate should be posible always 
        NC_points = np.array([sSr[:-1], np.log10(sSc)]).T
        crS_nh = np.array([NC_points[:-1], NC_points[1:]]).transpose(1, 0, 2)
        crS_nhlc = LineCollection(crS_nh, colors=col_sScnh[:-1], linewidth=1.5, label=r"$\Lambda_{net}$", linestyle="solid")
        ax4.add_collection(crS_nhlc)
        ax4.set_xlabel("Radius [kpc]", fontsize=12)
        ax4.set_ylim(-26,-20)
    ax4.set(xlim=(0,5))
    ax4.set_ylabel(r"$\Lambda_{net}$ [$log_{10}(erg \,\, cm^3 s^{-1})$]", fontsize=12)
    ax4.legend(loc='upper right', fontsize=12)

    ax5 = fig.add_subplot(2,3,5)
    plot_quant(radius[mask], metallicities[mask]/z_solar, ax5, prof_bins, 'mean', "Metallicity", "Metallicity", 'midnightblue')
    ax5.set_ylim(0, 4.0)

    ax6 = fig.add_subplot(2,3,6)
    plot_quant(radius[mask], xe[mask], ax6, prof_bins, 'mean', r'$\rm n_e/n_H$', r'$x_e$ (Electron Abundance)', 'midnightblue')
    ax6.set_ylim(0, 1.5)
    plt.tight_layout(w_pad=0.00, h_pad=0.00)

    # # SAVING THE IMAGES FOR TIMESTEP t 
    if FACE_ON: img_name = "cooling_face_t" + "%0.5f" % t
    else: img_name = "cooling_edge_t" + "%0.5f" % t
    print("generating image for time: ", str(t))
    print(output_directory + img_name + ".png")
    plt.savefig(output_directory + img_name + ".png", dpi=150, bbox_inches='tight') 

end = time.time()
print("elapsed time: ", end - start)
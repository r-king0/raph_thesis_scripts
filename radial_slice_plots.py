'''
    This file generates plots for density, energy, velocity, and temperature as the galactic disk as a function of a radial distance.
    The snapshots here provide a visual representation of the central edge of the disk of the galaxy. 

    Set up a sys argv for the run directory
    TODO: Fix rasterization
    TODO: Setup a utilities folder - load snapshots, making voronoi slices, and plots -> reduces the overall codebase size
'''

import h5py
import time 
import sys
import numpy as np    
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Arc
from matplotlib.ticker import FuncFormatter
from scipy import optimize
from scipy import stats
from scipy.spatial import cKDTree
import seaborn as sns
import pandas as pd 
mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

### PHYSICAL CONSTANTS ###
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS
z_solar = 0.02

#### Configuration Options ####
FACE_ON = False # Was predominantly used for examnining initial conditions and was rarely used for simulation analysis outside of diagnostics
T0_PLOT = True # Set to False if FACE_ON - pointless otherwise
CC85_PLOTS = True # Set to False if FACE_ON - pointless otherwise
EXTENDED = True
COOLING = True

simulation_directory = str(sys.argv[1]) 

################################
if FACE_ON: print("FACE_ON enabled.")

else: print("FACE_ON disabled. Output will be edge-on")

keys = ['Coordinates', 'Density', 'Velocities', 'InternalEnergy']
if COOLING: keys.append('ElectronAbundance')

data = {}
### PARAMETER CONSTANTS AND INITIAL VALUES ###
filename = "./snap_000.hdf5" 
with h5py.File(filename,'r') as f:
    parameters = dict(f['Parameters'].attrs)
    cells_per_dim = int(np.cbrt(len(f['PartType0']['Density'][()])))
    for key in keys:
        data[key] = f['PartType0'][key][()]
    header = dict(f['Header'].attrs)
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
    vel_x = data["Velocities"][:,0]
    vel_y = data["Velocities"][:,1] 
    vel_z = data["Velocities"][:,2] 
    if COOLING: abundance = data["ElectronAbundance"]
    else: abundance = 1

M_load = parameters["M_load"]
E_load = parameters["E_load"]
R = parameters["injection_radius"] # injection radius in kpc
sfr = parameters["sfr"]
UnitVelocity_in_cm_per_s = parameters["UnitVelocity_in_cm_per_s"] # 1 km/s
UnitLength_in_cm = parameters["UnitLength_in_cm"] # 1 kpc 
UnitMass_in_g = parameters["UnitMass_in_g"] # 1 solar mass
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.08568e+16 seconds 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.9889999999999999e+43 erg
UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3) # 6.76989801444063e-32 g/cm^3
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178297542e-22 barye
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS
InitDiskMetallicity = parameters["InitDiskMetallicity"]
boxsize = parameters["BoxSize"] # boxsize in kpc
inner_boxsize = 10
angle_l = 60
halfbox = boxsize/2
dx = inner_boxsize/cells_per_dim
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

### EQUATIONS ###
# Solution inside the injection radius
##### Taken from Chevalier and Clegg 85
def sol_in(M, r):
    T1 = ((3*gamma + 1/M**2)/(1+3*gamma))**(-(3*gamma+1)/(5*gamma+1))
    T2 = ((gamma - 1 + 2/M**2)/(1 + gamma))**((gamma+1)/(2*(5*gamma+1)))
    return T1*T2 - r/R

# Solution outside the injection radius
##### Taken from Chevalier and Clegg 85
def sol_out(M, r):
    T = ((gamma - 1 + 2/M**2)/(1 + gamma))**((gamma + 1)/(2*(gamma - 1)))
    result = M**(2/(gamma - 1))*T - (r/R)**2
    return result

# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

# The voronoi slice of a certain region(usually the mid-plane) along the z-axis.
# NOTE: previous versions used an interpolator that resulted in the code recreating itself at each stage. Newer versions instead use a tree. 
def make_voronoi_slice_edge(gas_xyz, gas_values, image_num_pixels, image_y_value, image_xz_max, tree): 
    s = image_xz_max/image_num_pixels
    xs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)
    zs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)
    X,Z = np.meshgrid(xs,zs)

    # NOTE: np.transpose(np.vstack(...)) is equivalent to np.column_stack()
    M_coords = np.column_stack([X.ravel(), np.full(len(X.ravel()), image_y_value), Z.ravel()])
    _, idx = tree.query(M_coords)
    result = np.transpose(gas_values[idx].reshape(len(xs), len(zs))) # rotate to main consistency

    return result, xs, zs # NOTE: np.array(xs) -> just xs, it's already an np array

# The voronoi slice of a certain region(usually the mid-plane) along the y-axis.
def make_voronoi_slice_face(gas_xyz, gas_values, image_num_pixels, image_z_value, image_xy_max, tree): 
    s = image_xy_max/image_num_pixels
    xs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)
    ys = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s) 
    X,Y = np.meshgrid(xs,ys)

    M_coords = np.column_stack([X.ravel(), Y.ravel(), np.full(len(X.ravel()), image_z_value)])
    _, idx = tree.query(M_coords)
    result = np.transpose(gas_values[idx].reshape(len(xs), len(ys))) # rotate to main consistency

    return result, np.array(xs), np.array(ys)

def plot_face(ax, coordinates, value, bins, center, boxsize, minimum, maximum, log, tree):
    stat, x_edge, y_edge = make_voronoi_slice_face(coordinates, value, bins, center, boxsize, tree)
    face_mesh = ax.pcolormesh(x_edge, y_edge, stat.T, shading='auto')
    if (log): face_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    else: face_mesh.set_clim(minimum, maximum)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    ax.set_xlabel('X [kpc]', fontsize=13)
    ax.set_ylabel('Y [kpc]', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)

def plot_edge(ax, coordinates, value, bins, center, boxsize, minimum, maximum, log, tree):
    stat, x_edge, z_edge = make_voronoi_slice_edge(coordinates, value, bins, center, boxsize, tree)
    edge_mesh = ax.pcolormesh(x_edge, z_edge, stat.T, shading='auto')
    if (log): edge_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    else: edge_mesh.set_clim(minimum, maximum)
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set_xlabel('X [kpc]', fontsize=13)
    ax.set_ylabel('Z [kpc]', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)

def make_cbar(plot_name, ax, pad, labeling, label_ticks, log):
    cbar = plt.colorbar(plot_name, ax = ax, pad=pad) 
    cbar.set_label(labeling, fontsize=13)
    cbar.set_ticks(label_ticks)
    if log:cbar.set_ticklabels([round(np.log10(label)) for label in (label_ticks)])

def quantity_plots(ax, coord, quantity, ang_quant, prof_bins, stat, upper_x, ylims, ylabeling, log):
    quantity_stat, r_edge, _ = stats.binned_statistic(coord, quantity, bins=prof_bins, statistic=stat, range=(0, upper_x))
    quant_ser = pd.Series(quantity_stat).ffill() # NOTE: we convert to a p.series to better fill in missing bins since the z-axis tends to not have as many values
    if log: ax.semilogy(r_edge[:-1], quant_ser, color='midnightblue', label = r"t = $%0.1f$ Myr" % times) 
    else: ax.plot(r_edge[:-1], quant_ser, label='t = $%0.1f$ Myr' % times, color='midnightblue') 
    
    if FACE_ON: ax.set_xlabel("Radial Distance [kpc]", fontsize=13)
    else:   
        quant_a, r_edge_a, _ = stats.binned_statistic(radius[angular_region], ang_quant[angular_region], bins=n_bins, statistic="median", range=(0, upper_x))
        if log: ax.semilogy(r_edge_a[:-1], quant_a, color='green', label = r"Bicone - t = $%0.1f$ Myr" % times) 
        else: ax.plot(r_edge_a[:-1], quant_a, color='green', label = r"Bicone - t = $%0.1f$ Myr" % times) 
        ax.set_xlabel("Radius [kpc]", fontsize=13)

    ax.set(xlim=(0, upper_x), ylim=ylims)    
    ax.set_ylabel(ylabeling, fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)

def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"

## ANALYTIC SOLUTION CALCULATION FOR COMPARISION - CHANGE NUMBERS AS NEEDED ###
if (CC85_PLOTS):
    r_an = np.linspace(0.001, boxsize, 1500)
    r_in = r_an[np.where(r_an <= R)]
    r_out = r_an[np.where(r_an > R)]

    s_in_yr = 3.154e+7
    grams_in_M_sun = 1.989e33
    M_dot_wind = sfr*M_load # solar masses per 1 year -> get this in grams per second 
    M_dot_cm = (M_dot_wind*UnitMass_in_g)/s_in_yr # grams/second
    E_dot_wind = E_load*3e41*sfr # this is in ergs/second 

    M_dot_code = M_dot_wind/(UnitMass_in_g/grams_in_M_sun)*(UnitTime_in_s/s_in_yr)
    E_dot_code = E_dot_wind/UnitEnergy_in_cgs*UnitTime_in_s

    M1 = optimize.fsolve(sol_in, x0=np.full(len(r_in), 0.001), args=(r_in))
    M2 = optimize.fsolve(sol_out, x0=np.full(len(r_out), 100), args=(r_out))
    M = np.concatenate([M1, M2])

    v_an = (M*np.sqrt(E_dot_code/M_dot_code)*(((gamma - 1)*M**2 + 2)/(2*(gamma - 1)))**(-0.5)) # this is in code units
    v_in = v_an[np.where(r_an <= R)]
    v_out = v_an[np.where(r_an > R)]
    cs = np.sqrt( (E_dot_code/M_dot_code)*(((gamma - 1)*M**2 + 2)/(2*(gamma - 1)))**(-1))
    cs_cm = cs*(UnitVelocity_in_cm_per_s) 

    rho_in = M_dot_code/(4*np.pi*v_in)*(r_in/R**3)*UnitDensity_in_cgs
    rho_out = M_dot_code/(4*np.pi*v_out)*1/r_out**2*UnitDensity_in_cgs
    rho_an = np.concatenate([rho_in, rho_out])
    rho_n = np.concatenate([rho_in, rho_out])/PROTON_MASS_GRAMS # rho/(proton mass)

    pressure_an = ((rho_an*cs_cm**2)/gamma)/kb # -> (g/cm^3* cm^2/s^2) -> p/kb 

    # P/kb = rho/(mean molecular weight * proton mass) * T = P/kb = rho/(proton mass) * 1/mean molecular weight * T
    ## T = pressure_an/(rho_n)* mean molecular weight
    temp_an = pressure_an/(rho_n)*(mean_molecular_weight(1)/PROTON_MASS_GRAMS) # keep the mean molecular mass the same. 

###### t = 0.0  values #######
if T0_PLOT:
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2)
    temperature = Temp_S(abundance, internal_energy)

    if FACE_ON: 
        mask = (z_coord >=lower_bound) & (z_coord <= upper_bound) # & (radial_coord <= inner_boxsize/2*np.sqrt(2))
        r_face = radial_coord[mask] 
        radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + eps) 
        tvx = vel_x - radial_velocity*rad_x/(radial_coord + eps)
        tvy = vel_y - radial_velocity*rad_y/(radial_coord + eps)
        tan_velocity = np.sqrt(tvx**2 + tvy**2)
    else: 
        mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) # & (radius <= inner_boxsize/2*np.sqrt(3))
        if EXTENDED: z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound)
        else: z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound) & (radius <= inner_boxsize/2*np.sqrt(3))
        radial_velocity_spherical = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + eps)
        r_z = radius[z_mask]
        density_z = density[z_mask]
        rv_z = radial_velocity_spherical[z_mask]
        temp_z = temperature[z_mask]

    # initial density profile
    if FACE_ON: rho, dist, _ = stats.binned_statistic(r_face, density[mask], bins=n_bins)
    else: rho, dist, _ = stats.binned_statistic(r_z, density_z, bins=n_bins)
    rho_init = np.column_stack((dist[:-1], rho*UnitNumberDensity) ) # initial density as a 2d array

    # initial velocity profiles
    if FACE_ON:
        v_r, r_v, _ = stats.binned_statistic(r_face, radial_velocity[mask], bins=n_bins) # radial velocity
        v_t, r_t, _ = stats.binned_statistic(r_face, tan_velocity[mask], bins=n_bins) # tangential or circular velocity 
        tv_init = np.column_stack( (r_t[:-1], v_t) ) # tangential velocity as a 2d array
    else:
        v_r, r_v, _ = stats.binned_statistic(r_z, rv_z, bins=n_bins) # radial velocity 
    vr_init = np.column_stack((r_v[:-1], v_r)) # radial velocity as a 2d array

    # initial temperature profiles
    if FACE_ON: T, r, _ = stats.binned_statistic(r_face, temperature[mask], bins=n_bins)
    else: T, r, _ = stats.binned_statistic(r_z, temp_z, bins=n_bins)
    T_init = np.column_stack( (r[:-1], T) )

######### SIMULATION DATA #########
start = time.time()
data = {}
for i in np.arange(90, 101): # select the snapshot range to go through
    filename = "./snap_%03d.hdf5" % i
    with h5py.File(filename,'r') as f:
        for key in keys:
            data[key] = f['PartType0'][key][()]
        header = dict(f['Header'].attrs)
    coord = data["Coordinates"]
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
    vel_x = data["Velocities"][:,0]
    vel_y = data["Velocities"][:,1] 
    vel_z = data["Velocities"][:,2] 
    if COOLING: abundance = data["ElectronAbundance"]
    else: abundance = 1
    temperature = Temp_S(abundance, internal_energy)
    number_density = density*UnitNumberDensity   

    t = header["Time"]
    times = t*1000

    ''' Get the radial distance of the box'''
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize

    # NOTE: Masking - Replace face_mask and edge_mask with just mask since we use one, we're never using the other
    radius = np.sqrt(rad_x**2 + rad_y**2+ rad_z**2)

    if FACE_ON: 
        mask = (z_coord >=lower_bound) & (z_coord <= upper_bound) # & (radial_coord <= inner_boxsize/2*np.sqrt(2))
        radial_coord = np.sqrt(rad_x**2 + rad_y**2) 
        radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + eps) 
        tvx, tvy = vel_x - radial_velocity*rad_x/(radial_coord+dx), vel_y - radial_velocity*rad_y/(radial_coord+dx)
        tan_velocity = np.sqrt(tvx**2 + tvy**2)

        r_coord = radial_coord[mask] 
        dens = density[mask]
        temps = temperature[mask]

    else:
        mask = (y_coord >= lower_bound) & (y_coord <= upper_bound) # & (radius <= inner_boxsize/2*np.sqrt(3))
        radial_velocity_spherical = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + eps)

        if EXTENDED: z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound)
        else: z_mask = (y_coord >= lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound) & (radius <= inner_boxsize/2*np.sqrt(3))
        dens = density[z_mask]
        rv_z = radial_velocity_spherical[z_mask]
        temps = temperature[z_mask]

        theta = np.arccos(np.abs(rad_z)/(radius + eps))*180/np.pi 
        angular_region = (np.abs(theta) <= 60) # Excludes anything with absolute angles greater than 60 
        r_coord = radius[z_mask]

    tree = cKDTree(coord[mask]) # coordinates change with each snapshot so keep it different

    ### PLOTS ###
    fig = plt.figure(figsize=(15,9)) # 20/12 = 15/9
    fig.set_rasterized(True)    
    ax1 = fig.add_subplot(2,3,1)
    if FACE_ON: plot_face(ax1, coord[mask], density[mask]*UnitNumberDensity, n_bins, halfbox, box_range, 1e-4, 10000, log=True, tree=tree)
    else: plot_edge(ax1, coord[mask], density[mask]*UnitNumberDensity, n_bins, halfbox, box_range, 1e-4, 1, log=True, tree=tree)
    density_mesh = ax1.collections[0]
    density_mesh.set_cmap("mako")
    cbar_d = make_cbar(density_mesh, ax1, 0.02, r'Density [log($\rm cm^{-3}$)]', [1e-5*(10**(x)) for x in range(1,6)], log=True)

    background_rect = patches.Rectangle((0, 0.80), width=1, height=0.2, color='black', alpha=0.25, transform=ax1.transAxes, fill=True)
    ax1.add_patch(background_rect)
    ax1.text(0.01, 0.96,"t = %0.3f Myr" % times, transform=ax1.transAxes, color="white", fontsize=12)
    ax1.text(0.01, 0.91,r'LMC/M82 Disk - CIE+PIE', transform=ax1.transAxes, color="white", fontsize=12)
    ax1.text(0.03, 0.87,r"- $\beta = $" + str(M_load) +  r", $\alpha =$" + str(E_load) + r", $Z_{disk}= " + str(int(InitDiskMetallicity/z_solar)) + r"Z_\odot$", transform=ax1.transAxes, color="white", fontsize=12)
    ax1.text(0.03, 0.81,r"- $\dot{M}_{SFR} =  " + str(int(sfr)) +  r"M_\odot \, yr^{-1}$, " + r" $R_{inject} =$ " + str(int(R*1000)) + r"pc", transform=ax1.transAxes, color="white", fontsize=12)
    
    # 2D VELOCITY CENTER VORONOI SLICE 
    ax2 = fig.add_subplot(2,3,2)
    if FACE_ON: plot_face(ax2, coord[mask], tan_velocity[mask], n_bins, halfbox, box_range, 0, 200, log=False, tree=tree)
    else: plot_edge(ax2, coord[mask], radial_velocity_spherical[mask], n_bins, halfbox, box_range, -100, 1400, log=False, tree=tree)
    velocity_mesh = ax2.collections[0]
    velocity_mesh.set_cmap("viridis")
    if FACE_ON: cbar_v = make_cbar(velocity_mesh, ax2, 0.02, r'Tangential Velocity [km/s]', [-50, 0, 50, 100, 150, 200], log=False)
    else: cbar_v = make_cbar(velocity_mesh, ax2, 0.02, r'Radial Velocity [km/s]', [200*x for x in range(0, 8)], log=False)

    # 2D TEMPERATURE CENTER VORONOI SLICE 
    ax3 = fig.add_subplot(2,3,3)
    if FACE_ON: plot_face(ax3, coord[mask], temperature[mask], n_bins, halfbox, box_range, 1e3, 1e7, log=True, tree=tree)
    else: plot_edge(ax3, coord[mask], temperature[mask], n_bins, halfbox, box_range, 1e3, 1e7, log=True, tree=tree)
    T_mesh = ax3.collections[0]
    T_mesh.set_cmap("plasma")
    cbar_T = make_cbar(T_mesh, ax3, 0.02, r'Temperature [log(K)]', [1*(10**(x)) for x in range(3,8)], log=True)

    ax3.axvline(50, linewidth=1, linestyle="dashed", color="white")
    ax3.axline((50, 50), slope=np.tan((90 - 60)*np.pi/180), linewidth=1, linestyle="dashed", color="white")
    ax3.axline((50, 50), slope=-np.tan((90 - 60)*np.pi/180), linewidth=1, linestyle="dashed", color="white")
    arc = Arc((50,50), width=3, height=3, angle=0, theta1=30, theta2=90, color="white", linestyle="dotted", linewidth=2)
    ax3.add_patch(arc)
    ax3.text(50 + 0.5, 50 + 1.8, "$60^{\circ}$", color="white", fontsize="medium")
    circle_bi = patches.Circle((50,50), radius=30, color="white", linestyle="solid", linewidth=2, fill=False)
    ax3.add_patch(circle_bi)

    # DENSITY RADIAL PROFILE
    ax4 = fig.add_subplot(2,3,4)
    quantity_plots(ax4, r_coord, dens*UnitNumberDensity, density*UnitNumberDensity, prof_bins, "median", upper_x, (1e-5,1e5), r"Density [$\rm cm^{-3}$]", log=True)
    if T0_PLOT: ax4.semilogy(rho_init[:,0], rho_init[:,1], label="t = 0.0 Myr", linestyle="dashed", color='midnightblue') 
    if CC85_PLOTS: ax4.plot(r_an, rho_n, label='CC85' % times, color='crimson') 
    ax4.legend(loc='upper right', fontsize=13)

    # VELOCITY RADIAL PROFILE 
    ax5 = fig.add_subplot(2,3,5)
    if FACE_ON: # deviates from the rest of the other plots and we're only using it once, so keep it seperate..
        v_r, r_v, _ = stats.binned_statistic(r_coord, radial_velocity[mask], bins=n_bins, range=(0, upper_x))
        v_t, r_t, _ = stats.binned_statistic(r_face, tan_velocity[mask], bins=n_bins, range=(0, upper_x))
        ax5.plot(r_v[:-1], v_r, label="Radial t = $%0.1f$ Myr" % times, color='midnightblue') 
        ax5.plot(r_t[:-1], v_t, label="Circular t = $%0.1f$ Myr" % times, color='red') #
        if T0_PLOT: ax5.plot(vr_init[:,0], vr_init[:,1], label='Radial t = 0.0 Myr', color='midnightblue', linestyle="dashed")
        if T0_PLOT: ax5.plot(tv_init[:,0], tv_init[:,1], label="Circular t = 0.0 Myr" % times, color='crimson', linestyle="dashed")
        ax5.set_xlabel("Radial Distance [kpc]", fontsize=13)
        ax5.set_ylabel("Velocity [km/s]", fontsize=13)
        ax5.set_ylim(-20, 220) # For a relaxed disk that is settled into equilibrium, the radial velocity should be around 0.
    else:
        quantity_plots(ax5, r_coord, rv_z, radial_velocity_spherical, prof_bins, "median", upper_x, (-10, 1500), "Radial Velocity [km/s]", log=False)
        if T0_PLOT: ax5.plot(vr_init[:,0], vr_init[:,1], label='t = 0.0 Myr', color='midnightblue', linestyle="dashed")
        if CC85_PLOTS: ax5.plot(r_an, v_an, label='CC85' % times, color='crimson') 
    ax5.legend(loc='upper right', fontsize=13)

    # TEMPERATURE RADIAL PROFILE
    ax6 = fig.add_subplot(2,3,6) 
    quantity_plots(ax6, r_coord, temps, temperature, prof_bins, "median", upper_x, (1e3,1e8), "Temperature [K]", log=True)
    if T0_PLOT: ax6.semilogy(T_init[:,0], T_init[:,1], label='t = 0.0 Myr', color='midnightblue', linestyle="dashed")
    if CC85_PLOTS: ax6.semilogy(r_an, temp_an, label='CC85', color='red') 
    ax6.legend(loc="upper right", fontsize=13)

    plt.tight_layout(w_pad=0.00, h_pad=0.00)
    # SAVING THE IMAGES FOR TIMESTEP t 
    if EXTENDED:
        if FACE_ON: img_name = "extended_face_t" + "%0.5f" % t
        else: img_name = "extended_edge_t" + "%0.5f" % t
    else:
        if FACE_ON: img_name = "face_t" + "%0.5f" % t
        else: img_name = "edge_t" + "%0.5f" % t
    print("generating image for time: ", str(t))
    plt.savefig(simulation_directory + img_name + ".png", dpi=150, bbox_inches='tight') 

end = time.time()
print("elapsed time: ", end - start)
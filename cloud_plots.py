# Loading libraries and key coordinates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import interpolate
from scipy import stats
import cmasher as cmr # requires Matplotlib 3.5 or newer
import seaborn as sns
import sys
plt.style.use("seaborn-v0_8-bright")

simulation_directory = str(sys.argv[1]) 

################################################
### IMPORTANT FUNCTIONS ### 
from utilities import load_snapshots, Temp_S
### PHYSICAL CONSTANTS ###
from utilities import PROTON_MASS_GRAMS, kb
##### UNITS #####
from utilities import UnitLength_in_cm, UnitMass_in_g, UnitTime_in_s, UnitEnergy_in_cgs, UnitDensity_in_cgs, UnitPressure_in_cgs
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178294542e-22 barye
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS
UnitEnergyDensity = UnitEnergy_in_cgs/pow(UnitLength_in_cm, 3)
T_COLD_MAX = 3e4

#### FUNCTIONS ####
def get_interior(rx, ry, rz, center_boxsize):
    return (np.abs(rx) <= center_boxsize) & (np.abs(ry) <=  center_boxsize) & (np.abs(rz) <=  center_boxsize) 

keys = ["Masses", "Coordinates", "Pressure", "Density", "Velocities","ElectronAbundance", "InternalEnergy"]

###### KEY SIMULATION PARAMETERS ######
boxsize = 100
midpoint = boxsize/2 
center_boxsize = 10
dx = center_boxsize/300
eps = dx/1e6

filename = "/vera/ptmp/gc/raki/outflow_cooling/output_PIE_fid/snap_100.hdf5"
data, header, parameters = load_snapshots(filename, keys)

R = parameters["injection_radius"]
boxsize = parameters["BoxSize"]
coordinates = data["Coordinates"]
x_coord = data["Coordinates"][:,0] 
y_coord = data["Coordinates"][:,1]
z_coord = data["Coordinates"][:,2]
density = data["Density"]
masses = data["Masses"]
x_e = data["ElectronAbundance"]
internal_energy = data["InternalEnergy"]
vel_x = data["Velocities"][:,0]
vel_y = data["Velocities"][:,1] 
vel_z = data["Velocities"][:,2] 

time = header["Time"]*1000
temperature = Temp_S(x_e, internal_energy)
rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 
number_densities = density*UnitNumberDensity
inner_box = get_interior(rad_x, rad_y, rad_z, center_boxsize)
pressures = data["Pressure"] 
pressure_cgs = pressures*UnitPressure_in_cgs/kb

theta = np.arccos(np.abs(rad_z)/(radius + eps))*180/np.pi 
# cloud is somewhere between:
x_cm, x_cmax = 0.8, 1.7
z_cm, z_cmax = 1.3, 2.2

xlimits = (0.0, 3.5)
zlimits = (0.5, 4)

cloud_quantity = ["Temperature [K]", "Radial Velocity [km/s]",r"Pressure [$\rm K cm^{-3}$]"]
cmaps = ["inferno", "crest_r", "viridis"]
limits = [(1e3, 1e6), (400, 1200), (1e2, 1e5)]
rbins = np.linspace(2.2, 4.0, 12)
xticks_vals = [[2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9], [2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9], [1,2,3,4,5]]

y_locate_mask = ((rad_x > x_cm) & (rad_x < x_cmax) & 
                 (rad_z > z_cm) & (rad_z < z_cmax) &  
                 (inner_box) 
                )

radial_velocity = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + eps)
cold_cloud_mask = (y_locate_mask) & (temperature <= T_COLD_MAX)
x_mesh = rad_x[cold_cloud_mask] 
y_mesh = rad_y[cold_cloud_mask]  
z_mesh = rad_z[cold_cloud_mask]
masses_y = masses[cold_cloud_mask] # weighted by the masses 

x_cloud_center = np.average(x_mesh, weights=masses_y)
y_cloud_center = np.average(y_mesh, weights=masses_y)# so the weighted number that contains where the center of y is.
z_cloud_center = np.average(z_mesh, weights=masses_y)
print(f"cloud center: r_x = %0.2f, r_y = %0.2f, r_z = %0.2f" % (x_cloud_center, y_cloud_center, z_cloud_center))
cloud_variance = np.sqrt(np.average((y_cloud_center - y_mesh)**2, weights=masses_y))
print("cloud variance:", cloud_variance) # the cloud center of mass is ~ 1.42 +- 0.58

cloud_mask =  (
                (rad_y >= y_cloud_center - (cloud_variance*2))   & (rad_y <= y_cloud_center + (cloud_variance*2))  
                & (rad_x >= xlimits[0] ) & (rad_x <= xlimits[1] )
                & (rad_z >= zlimits[0]) & (rad_z <= zlimits[1] )
              )

angular_region = (np.abs(theta) <= 60) # Excludes anything with absolute angles greater than 60 
cone_angle_mask = (np.abs(theta) >= 40) & (np.abs(theta) <= 60) 

t_l = 1e5  # ~t_l = np.percentile(temperature[rad_z >= 0.5], 0.15) 
nd_h = 0.0065 # nd_h = np.percentile(number_densities[rad_z >= 0.5], 99.85)  from snapshot 000
nh_c = np.max(number_densities[(angular_region) & (temperature >=  2*T_COLD_MAX)]) # highest density that that's from not from a cloud or a disk
bg_cells = (number_densities <= nd_h) & (temperature >= t_l) & (np.abs(radial_velocity) <= 40) # Gets rids of as much bg cells  BG cells.
mean_number_density = np.mean(number_densities[(angular_region) & (radius >= R*2) & (number_densities < nh_c) & (radius < 30) & (~bg_cells)]) # to prevent any potential cool clouds& (~bg_cells)]) 
overdensity = (number_densities - mean_number_density)/mean_number_density
bicone_cloud = cloud_mask & inner_box & cone_angle_mask  & (overdensity >= 5) & (temperature <= T_COLD_MAX)
bicone_mixing = cloud_mask & inner_box & cone_angle_mask & (temperature >= T_COLD_MAX) & (temperature <= T_COLD_MAX*10) # & (overdensity >= 5) & (overdensity >= 1)  # & (overdensity > 1) & (overdensity <= 10)
print("mean number density:", mean_number_density)

cloud_mask_vor =  (
                (rad_y >= y_cloud_center - (dx*12))   & (rad_y <= y_cloud_center + (dx*12)) 
                & (rad_x >= xlimits[0] ) & (rad_x <= xlimits[1] )
                & (rad_z >= zlimits[0]) & (rad_z <= zlimits[1] )
            )
cloud_coords = np.column_stack([rad_x[cloud_mask_vor], rad_z[cloud_mask_vor]])

x_grid = np.linspace(xlimits[0], xlimits[1], 2000)
z_grid = np.linspace(zlimits[0], zlimits[1], 2000)
GX, GZ = np.meshgrid(x_grid,z_grid)
flattened_grid = np.column_stack([GX.ravel(), GZ.ravel()])

for i, quant in enumerate(cloud_quantity):
    fig = plt.figure(figsize=(10, 4.0))
    fig.set_rasterized(True)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    if quant != r"Pressure [$\rm K cm^{-3}$]": inset_plot = ax1.inset_axes([0.1, 0.1, 0.86, 0.20], sharex=ax1)
    else: inset_plot = ax1.inset_axes([0.1, 0.1, 0.86, 0.20])

    if quant == "Temperature [K]":
        q_values = temperature[cloud_mask_vor]
        T_cloud, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], temperature[bicone_cloud], bins=rbins, statistic="median")
        T_mixing, r_edge, _ = stats.binned_statistic(radius[bicone_mixing], temperature[bicone_mixing], bins=rbins, statistic="median")
        r_faces = 0.5* (r_edge[:-1] + r_edge[1:])
        ax1.semilogy(r_faces, T_mixing, color="black", linestyle="dotted", label="Mixing")
        ax1.semilogy(r_faces, T_cloud, color="black", label="Cloud")

        inset_plot.plot(r_faces, T_cloud/T_mixing, color="black")
        inset_plot.set_ylim(0.0, 0.2)
        inset_plot.set_title(r"$\rm T_{cl}/T_{m}$")

        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.set_ylabel(r"Temperature  [K]", fontsize=13)
        ax1.set(xlim=(2.5, 3.9), ylim=(1e2, 1e6))

    elif quant == "Radial Velocity [km/s]":
        q_values = radial_velocity[cloud_mask_vor]

        v_cloud, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], radial_velocity[bicone_cloud], bins=rbins, statistic="median")
        v_mixing, r_edge, _ = stats.binned_statistic(radius[bicone_mixing], radial_velocity[bicone_mixing], bins=rbins, statistic="median")
        r_faces = 0.5* (r_edge[:-1] + r_edge[1:])

        rv_wind, rw_edge, _ = stats.binned_statistic(radius[angular_region], radial_velocity[angular_region], bins=rbins, statistic="median")
        ax1.plot(rw_edge[:-1], rv_wind, linestyle="solid", label="Wind (Median)", color="darkorange")

        rv_cloud, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], radial_velocity[bicone_cloud], bins=rbins, statistic="median")
        r_faces = 0.5* (r_edge[:-1] + r_edge[1:])

        ax1.plot(r_faces, rv_cloud, color="black", label="Cloud")
        ax1.set_ylabel(r"Radial Velocity [$\rm km/s$]", fontsize=13)

        ax1.plot(r_faces, v_mixing, color="black", linestyle="dashed", label="Mixing")

        inset_plot.plot(r_faces, v_mixing/rv_wind, color="black",linestyle="dashed")
        inset_plot.plot(r_faces, rv_cloud/rv_wind, color="black")
        inset_plot.set_ylim(0.0, 1.1)
        inset_plot.set_yticks([0, 0.5, 1])
        inset_plot.set_title(r"$\rm v/v_{wind}$")

        ax1.set(xlim=(2.3, 3.9), ylim=(-200, 1600))

    elif quant == r"Pressure [$\rm K cm^{-3}$]":
        q_values = pressure_cgs[cloud_mask_vor]
        p_bins = np.linspace(1, 5, 200) # uses.a different binning
        p_cloud, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], pressure_cgs[bicone_cloud], statistic="median", bins=p_bins)
        p_cloud_high, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], pressure_cgs[bicone_cloud],  statistic="max", bins=p_bins)
        p_cloud_low, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], pressure_cgs[bicone_cloud],  statistic="min", bins=p_bins)
        r_faces = 0.5* (r_edge[:-1] + r_edge[1:])

        p_wind, rwind, _ = stats.binned_statistic(radius[angular_region], pressure_cgs[angular_region], bins=p_bins, statistic="median", range=(1, 5))
        rw_faces = 0.5* (rwind[:-1] + rwind[1:])

        ax1.semilogy(rw_faces, p_wind, color="black", label="Wind")
        ax1.semilogy(r_faces, p_cloud, color="royalblue", label="Cloud")

        ax1.set_yticks([10**x for x in np.arange(0, 6)])
        ax1.set_ylabel(r"Pressure [$\rm K \, cm^{-3}$]", fontsize=13)

        inset_plot.plot(r_faces, p_cloud/p_wind, color="royalblue")
        inset_plot.axhline(1, linestyle="dashed", color="black", linewidth=1)
        inset_plot.set_ylim(0.0, 3)
        inset_plot.set_yticks([0, 1, 2, 3])
        inset_plot.set_xlim(1.8, 3.5)
        inset_plot.set_title(r"$\rm P_{cl}/P_w$")

        ax1.set(xlim=(1, 5), ylim=(1, 1e5))

    inset_plot.tick_params(axis='x', bottom=True, labelsize=9)
    inset_plot.tick_params(axis='y', bottom=True, labelsize=9)
    
    ax1.set_xticks(xticks_vals[i])

    ax1.set_xlabel("Radius [kpc]", fontsize=13)
    ax1.legend(loc="upper right", fontsize=12)
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)


    interp = interpolate.NearestNDInterpolator(cloud_coords, q_values)
    result = np.transpose(interp(flattened_grid).reshape(len(x_grid), len(z_grid)))
    if quant != "Radial Velocity [km/s]": edge_mesh = ax2.pcolormesh(GX, GZ, result.T, shading='auto', cmap=cmaps[i], norm=colors.LogNorm(limits[i][0], limits[i][1]))
    else: edge_mesh = ax2.pcolormesh(GX, GZ, result.T, shading='auto', cmap=cmaps[i], norm=colors.Normalize(limits[i][0], limits[i][1]))

    cbar = plt.colorbar(edge_mesh, ax = ax2, pad=0.005)
    cbar.set_label(quant, fontsize=13)

    ax2.set(xlim=(0, 3.5), ylim=(0.5, 4.0))

    ax2.set_xticks([0.0, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5])
    ax2.set_yticks([0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0])

    ax2.xaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

    ax2.set_xlabel('X [kpc]', fontsize=13)
    ax2.set_ylabel('Z [kpc]', fontsize=13)

    plt.subplots_adjust(wspace=0.2)
    plt.savefig(f"cloud_{quant.split('[')[0].strip()}_PIE_M82.pdf", dpi=150, bbox_inches='tight') # use this to get the mean velocity..
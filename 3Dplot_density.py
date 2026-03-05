import numpy as np
from simext3d import *
from matplotlib import pyplot as plt
from my_14_numbers import *
import cmasher as cmr
import sys
import pandas
import my_matplotlib_style
from photosphere import *
from my_14_numbers import *
from mpl_toolkits.mplot3d import Axes3D
#==========Data load====================
def main(d):
    sim = simext3d(d, tracer=True)
    sim.sort_scalar2grid('r')
    sim.sort_scalar2grid('th')
    sim.sort_scalar2grid('ph')
    sim.sort_scalar2grid('rho')
    sim.sort_scalar2grid('bsq')
    sim.sort_scalar2grid('tracer')
    sim.sort_scalar2grid('uint')
    sim.sort_vector2grid('ucon')
    
    sim.sort_tensor2grid('Rmunu')

 #============Computation=====================   
    density = sim.rho_grid*rhocgs
    sim.tau_grid = calc_theta_tausca_withalldamping(sim, gmc2, rhocgs,kappa_sca)
    print("th_grid:", sim.th_grid.min(), sim.th_grid.max(), 
          "ph_grid:", sim.ph_grid.min(), sim.ph_grid.max())
    # exit()
# pick the phi slice closest to φ = 0
    def find_nearest_index(arr, value):
        return np.argmin(np.abs(arr - value))

    w = find_nearest_index(sim.ph_grid[0, 0, :], 0)
    # print(f"Selected φ index: {w}, φ value: {sim.ph_grid[0, 0, w]}")
    # q = find_nearest_index(sim.th_grid[0,:,0],1.57)
    # print(f"Selected θ index: {q}, θ value: {sim.th_grid[0, q, 0]}")
    # p = find_nearest_index(sim.r_grid[:,0,0],50) 
    # print(f"Selected r index: {p}, r value: {sim.r_grid[p, 0, 0]}")

# take slice in φ = 0 plane (meridional plane)
    r = sim.r_grid
    theta = sim.th_grid
    phi = sim.ph_grid
    density = density
    sim.tau_grid = sim.tau_grid 
# Spherical to Cartesian coordinates
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    
    # print("x:",x.min(), x.max(), "y:", y.min(), y.max(), "z:", z.min(), z.max())
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(y, x,z, c=np.log10(density), cmap='viridis',vmax=-3, vmin=-18, s=3, alpha=0.12)
    # ax.plot_surface(y[30, :, :], x[30, :, :], z[30, :, :], 
    #                 facecolors=plt.cm.viridis(np.clip((np.log10(density[30, :, :]) + 10) / 7, 0, 1)),
    #                 shade=False, alpha=0.6)
    cbar = fig.colorbar(scatter, ax=ax, label=r'$log_{10} \rho\,[\mathrm{g}\, \mathrm{cm}^{-3}]$', shrink=.5, aspect=35)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1.5)
    ax.set_xlabel(r'$x\,[GM/c^2]$')
    ax.set_zlabel(r'$z\,[GM/c^2]$')
    ax.set_ylabel(r'$y\,[GM/c^2]$')

    plt.savefig("test_3d.png", dpi=300, bbox_inches='tight')
    # plt.show()
    # print(z[:,0])
    # print(np.shape(x), np.shape(y), np.shape(z), np.shape(density))
    # exit()

    
#========Contour===================
    # ax.contour(x, z ,sim.tau_grid , levels=[1],colors='grey', linestyles='--', linewidths=1.8)


if __name__ == '__main__':
    main(sys.argv[1])


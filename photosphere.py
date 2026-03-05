import numpy as np
from scipy.integrate import cumtrapz

def adjust_below_rlc(r_grid, rcut=55., rns=5.):
    adjustment = np.where(r_grid>rcut,0.,
                np.where(r_grid<=rns, 1.,
                    1./(1.+.1*np.exp(r_grid-38.))))
    return adjustment
#check!!!!
kappa_sca = 0.34
def calc_theta_tausca(simext, gmc2, rhocgs, kappa_sca):
    integrand = 10**(-8*simext.tracer_grid)*simext.rho_grid*kappa_sca*gmc2*rhocgs*simext.r_grid
    upper = cumtrapz(integrand, simext.th_grid, axis=1, initial=0.)
    lower = np.flip(cumtrapz(np.flip(integrand,1), simext.th_grid, axis=1,initial=0.),1)
    return np.where(simext.th_grid>np.pi/2.,lower,upper)

def calc_theta_tausca_nodamping(simext, gmc2, rhocgs, kappa_sca):
    integrand = simext.rho_grid*kappa_sca*gmc2*rhocgs*simext.r_grid
    upper = cumtrapz(integrand, simext.th_grid, axis=1, initial=0.)
    lower = np.flip(cumtrapz(np.flip(integrand,1), simext.th_grid, axis=1,initial=0.),1)
    return np.where(simext.th_grid>np.pi/2.,lower,upper)

def calc_theta_tausca_withalldamping(simext, gmc2, rhocgs, kappa_sca):
    integrand = 10**(-8*simext.tracer_grid*adjust_below_rlc(simext.r_grid))*simext.rho_grid*kappa_sca*gmc2*rhocgs*simext.r_grid
    upper = cumtrapz(integrand, simext.th_grid, axis=1, initial=0.)
    lower = np.flip(cumtrapz(np.flip(integrand,1), simext.th_grid, axis=1,initial=0.),1)
    return np.where(simext.th_grid>np.pi/2.,lower,upper)

def calc_theta_tausca_upalldamping(simext, gmc2, rhocgs, kappa_sca):
    integrand = 10**(-8*simext.tracer_grid*adjust_below_rlc(simext.r_grid))*simext.rho_grid*kappa_sca*gmc2*rhocgs*simext.r_grid
    upper = cumtrapz(integrand, simext.th_grid, axis=1, initial=0.)
    # lower = np.flip(cumtrapz(np.flip(integrand,1), simext.th_grid, axis=1,initial=0.),1)
    return upper

def calc_drcell_tausca(simext, gmc2, rhocgs, kappa_sca):
    r = simext.r_grid
    dr_grid = np.full(r.shape,0.)
    dr_grid[1:,:] = r[1:,:] - r[:-1,:]

    tau = dr_grid*gmc2*simext.rho_grid*rhocgs*kappa_sca
    return tau


def calc_tausca_withdamp(simext, gmc2, rhocgs, kappa_sca):
    r_mask = 1000
    masked_rho = np.where(simext.r_grid>r_mask,0.,simext.rho_grid)
    flipped_r = np.flip(simext.r_grid,0)
    flipped_rho = np.flip(masked_rho,0)
    flipped_tracer = np.flip(simext.tracer_grid,0)
    #integrand = ((10**(-8*simext.tracer_grid*adjust_below_rlc(simext.r_grid)))*
    #        flipped_rho*kappa_sca*gmc2*rhocgs)
    integrand = ((10**(-8*flipped_tracer*adjust_below_rlc(flipped_r)))*
            flipped_rho*kappa_sca*gmc2*rhocgs)
    #integrand = ((10**(-8*flipped_tracer))*
    #        flipped_rho*kappa_sca*gmc2*rhocgs)
    #integrand = (1.-simext.tracer_grid)*flipped_rho*kappa_sca*gmc2*rhocgs
    flipped_tausca_grid = -cumtrapz(integrand, flipped_r, axis=0,initial=0)
    return np.flip(flipped_tausca_grid,0)

def extended_tau_r(simext, rmax, gmc2, rhocgs, kappa_sca):
    test_i = 10
    test_rho = simext.rho_grid[:,test_i]
    r_line = simext.r_grid[:,test_i]
    plt.loglog(r_line, test_rho)
    plt.show()


def calc_tausca_withalldamping(simext, gmc2, rhocgs, kappa_sca):
    r_mask = 1000
    masked_rho = np.where(simext.r_grid>r_mask,0.,simext.rho_grid)
    flipped_r = np.flip(simext.r_grid,0)
    flipped_rho = np.flip(masked_rho,0)
    flipped_tracer = np.flip(simext.tracer_grid,0)
    #integrand = ((10**(-8*simext.tracer_grid*adjust_below_rlc(simext.r_grid)))*
    #        flipped_rho*kappa_sca*gmc2*rhocgs)
    integrand = ((10**(-8*flipped_tracer*adjust_below_rlc(flipped_r)))*
            flipped_rho*kappa_sca*gmc2*rhocgs)
    #integrand = ((10**(-8*flipped_tracer))*
    #        flipped_rho*kappa_sca*gmc2*rhocgs)
    #integrand = (1.-simext.tracer_grid)*flipped_rho*kappa_sca*gmc2*rhocgs
    flipped_tausca_grid = -cumtrapz(integrand, flipped_r, axis=0,initial=0)
    return np.flip(flipped_tausca_grid,0)

def calc_tausca_nodamping(simext, gmc2, rhocgs, kappa_sca):
    flipped_r = np.flip(simext.r_grid,0)
    flipped_rho = np.flip(simext.rho_grid,0)
    integrand = (flipped_rho*kappa_sca*gmc2*rhocgs)
    flipped_tausca_grid = -cumtrapz(integrand, flipped_r, axis=0,initial=0)
    return np.flip(flipped_tausca_grid,0)

def calc_tausca(simext, gmc2, rhocgs, kappa_sca):
    r_mask = 1000
    masked_rho = np.where(simext.r_grid>r_mask,0.,simext.rho_grid)
    flipped_r = np.flip(simext.r_grid,0)
    flipped_rho = np.flip(masked_rho,0)
    flipped_tracer = np.flip(simext.tracer_grid,0)
    #integrand = ((10**(-8*simext.tracer_grid*adjust_below_rlc(simext.r_grid)))*
    #        flipped_rho*kappa_sca*gmc2*rhocgs)
    #integrand = ((10**(-8*flipped_tracer*adjust_below_rlc(flipped_r)))*
    #        flipped_rho*kappa_sca*gmc2*rhocgs)
    integrand = ((10**(-8*flipped_tracer))*
            flipped_rho*kappa_sca*gmc2*rhocgs)
    #integrand = (1.-simext.tracer_grid)*flipped_rho*kappa_sca*gmc2*rhocgs
    flipped_tausca_grid = -cumtrapz(integrand, flipped_r, axis=0,initial=0)
    return np.flip(flipped_tausca_grid,0)

if __name__ == '__main__':
    import sys
    from my_14_numbers import *
    from simext2d import simext2d
    from matplotlib import pyplot as plt
    data = simext2d(sys.argv[1],tracer=True)
    data.sort_scalar2grid('rho')
    data.sort_scalar2grid('r')
    extended_tau_r(data,2000., gmc2, rhocgs)

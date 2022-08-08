import sys, os, time, fileinput
import numpy as np
import math as math 
from scipy.stats import maxwell, rv_histogram
from scipy.integrate import quad, dblquad
from scipy.special import erf
from scipy.optimize import minimize
import emcee

import pandas as pd

# ----------------------------------------

# input parameters

## perturber mass (M_sol)
log_m = float(sys.argv[1])
str_log_m = str.format('{0:.3f}',log_m)
m = 10**float(str_log_m)

## scale radius 
log_rs = float(sys.argv[2]) # NFW scale radius (pc)
str_log_rs = str.format('{0:.3f}',log_rs)

## power law index
alpha = float(sys.argv[3]) # power law index
str_alpha = str.format('{0:.3f}',alpha)

## halo fraction
log_f = float(sys.argv[4]) # halo fraction [setting max number of time steps]
f = 10**log_f
str_log_f = str.format('{0:.3f}',log_f)

# threshold fractional energy injection defining p_max
log_eps = float(sys.argv[5])  
energy_fraction = 10**log_eps
str_log_eps = str.format('{0:.2f}',log_eps)

# ------------------------
## approximate number of encounters
log_K = float(sys.argv[6])    # log number of samples
str_log_K = str.format('{0:.2f}',log_K)
K = int(10**log_K)
# ------------------------
## list of a0-values
bin_scale = sys.argv[7]

log_alow = float(sys.argv[8])
str_log_alow = str.format('{0:.3f}',log_alow)
log_ahigh = float(sys.argv[9])
str_log_ahigh = str.format('{0:.3f}',log_ahigh)
step_size = float(sys.argv[10]) # number of bins
str_step_size = str(step_size)
#str_step_size = str.format('{0:.3f}',step_size)

log_alow_offset = float(sys.argv[11])
str_log_alow_offset = str.format('{0:.3f}',log_alow_offset)
log_ahigh_offset = float(sys.argv[12])
str_log_ahigh_offset = str.format('{0:.3f}',log_ahigh_offset)
#-------------------------

## Conversion Factors:
kg_g = 1E-3
Msolar_kg = 5.02785E-31 # Msolar/kg
pc_m = 3.24078E-17 # pc/m
Gyr_yr = pow(10,-9) # Gyr/yr
yr_s = 3.17098E-8 # yr/s
km_m = 1/1000 # km/m
kg_GeV = 1.78266E-27 # [kg/(GeV/c**2)]
m_cm = pow(10,-2) # m/cm
pc_AU = 4.84814e-6 # pc/AU
AU_pc = 206265  # AU/pc

#-------------------------
## fixed parameters 
#m1 = 0.5 # mass of primary (M_sol)
#m2 = 0.5 # mass of secondary (M_sol)
v0 = 240 # circular velocity (km/s) 
vesc = 533 # local escape velocity (km/s)
T = 10 # evolution time (Gyr)
#rho = 0.009 # Local DM Density (M_sol / pc**3) [Yoo]
rho_pdg = 0.39 # Local DM Density (GeV / cm3)
rho = 0.39 * (Msolar_kg) * (kg_GeV) * (pc_m)**(-3) * (m_cm)**(-3) #[Ed]


## constants
G = 4.30091e-3   # Grav Constant ([pc (M_solar)^-1 (km/s)**2])

## set up maximum impact parameter and time steps

### Compute the average inverse velocity for the truncated velocity profile

def v_theta_(cos_theta,v,vesc):
    return -v*cos_theta + np.sqrt(v*v*cos_theta*cos_theta + (vesc*vesc - v*v))

def fv_no_vesc_(v, sigma):
    return v*v*np.exp(-v*v / 4 / sigma / sigma)

def dfv_dcos_theta_eff_(v_theta,sigma):
    return (1/4) * (-2 * v_theta * np.exp( - v_theta * v_theta / sigma / sigma ) 
                 + np.sqrt(np.pi) * sigma * erf(v_theta / sigma) ) 

def inv_v_avg_(sigma,vesc,normalization):
    v_avg_eff = ( dblquad(lambda cos_theta, v: 
                                     fv_no_vesc_(v, sigma) * dfv_dcos_theta_eff_(v_theta_(cos_theta,v/2,vesc),sigma) / v, 0, 2*vesc, 
                                     lambda v: 0, lambda v: 1)[0] )
    return v_avg_eff / normalization

sigma = v0/math.sqrt(2)/math.sqrt(2)
normalization = ( dblquad(lambda cos_theta, v: 
                                     fv_no_vesc_(v, sigma) * dfv_dcos_theta_eff_(v_theta_(cos_theta,v/2,vesc),sigma), 0, 2*vesc, 
                                     lambda v: 0, lambda v: 1)[0] )
inv_v_avg = inv_v_avg_(sigma,vesc,normalization)

### max impact parameter when the fractional energy injection falls below energy_fraction
def p_max_(a,m,M,inv_v_avg,f,energy_fraction): 
    bar_C = 2*G*m * 2*G*m / 2 * inv_v_avg
    E = G * M / 2 / a
    inv_factor = np.pi * (f * rho * T / m) * bar_C / E
    bar_delta_Eps_val = inv_factor**(-1) * energy_fraction
    #if m >= 10:
    #    return a * np.sqrt(1 + 2 * bar_delta_Eps_val**(-1) * (1 + np.sqrt(1 + bar_delta_Eps_val))) # pc
    #else:
    #    return (m/10)**(1/2) * a * np.sqrt(1 + 2 * bar_delta_Eps_val**(-1) * (1 + np.sqrt(1 + bar_delta_Eps_val))) # pc
    return a * np.sqrt(1 + 2 * bar_delta_Eps_val**(-1) * (1 + np.sqrt(1 + bar_delta_Eps_val))) # pc

### load number of time steps needed to evolve for the specified evolution time
def delta_t_(m,inv_v_avg,f,R):    # length of time step in Gyr
    delta_t_val = m / (f * rho * np.pi * R**2) * inv_v_avg
    delta_t_val_Gyr = delta_t_val * ( (km_m) / (pc_m) ) * ( (Gyr_yr) * (yr_s) ) 
    return delta_t_val_Gyr  # Gyr

p_max = p_max_(1,m,1,inv_v_avg,1,energy_fraction) # p_max value   # p_max value 
delta_t = delta_t_(m,inv_v_avg,f,p_max)    # delta_t value
N = int(T / delta_t)    # number of time steps

# ------------------------------------------------------

# load samples
## parameters
N_enc = int(K*N) # number of samples
R = p_max   # maximum impact parameter (pc)

# ----------------------------------------
## Load energy injection rate data
local_dir = '/het/p1/ramirez/dmbinaries/shooter_edr3/newgen_pipeline/sep_dist/'
sorted_dir = local_dir + 'sorted_data/' +str_log_m+'_'+str_log_rs+'_'+str_alpha+'_'+str_log_f+'_'+str_log_eps+'_'+str_log_K
file_name = 'hists_'+str_log_m+'_'+str_log_rs+'_'+str_alpha+'_'+str_log_f+'_'+str_log_eps+'_'+str_log_K+'_'+bin_scale+'_'+str_log_alow+'_'+str_log_ahigh+'_'+str_step_size+'_'+str_log_alow_offset+'_'+str_log_ahigh_offset

results_list = np.load(os.path.join(sorted_dir, file_name + '.npy'), allow_pickle = True)

# ----------------------------------------
## Load a-values and bins

# make the array of semimajor axis values
if bin_scale=='log': 
    bins = np.arange(log_alow+log_alow_offset,log_ahigh+log_ahigh_offset+step_size,step_size)  # bins
    bins = 10**bins
    mid = 0.5 * (bins[1:] + bins[:-1])

if bin_scale=='lin':
    alow=10**(log_alow + log_alow_offset)
    ahigh=10**(log_ahigh + log_ahigh_offset)

    Nbins = 180
    bins = np.arange(alow,ahigh,Nbins)  # bins
    mid = 0.5 * (bins[1:] + bins[:-1])

# ----------------------------------------
## Load catalog (updated for loading eDR3 halo subsample)
data_dir = '/het/p4/ramirez/dmbinaries/shooter_edr3/newgen_pipeline/sep_dist/' + 'data/'   # LOCAL loading

# ----------------------------------------
## Load catalog (updated for loading eDR3 halo subsample)

catalog_name = data_dir + 'catalog_data' + '_' + '1.0' + '_' + '1.0' + '_cut_err' + '.npy'
catalog_sep_vec, catalog_dist_vec, catalog_delta_G_vec, catalog_R_chance_align_vec = np.load(catalog_name, allow_pickle = True)

# ----------------------------------------
# This block was added in the contamination version: 
# Contamination probability extrapolated to > 1 past the s > 1 pc.
# This requires us to modify the selection fct matrix AND the log-likelihood function
s_min = np.min(catalog_sep_vec)
s_max = np.max(catalog_sep_vec)
n_min = np.where(bins <= s_min)[0][-1]
n_max = np.where(bins >= s_max)[0][0]

bins_data = bins[n_min:n_max+1]
mid_data = 0.5 * (bins_data[1:] + bins_data[:-1])

## Preload selection function matrix

from scipy.interpolate import interp1d

def selection_fct(theta,theta0):
    if theta < theta0:
        return 0
    else:
        return 1
selection_fct_vec = np.vectorize(selection_fct)

delta_G_mid = [0.5,1.5,2.5,3.5,5,7,9.5]
theta_cut_vals = [3.06481262,  3.26382633,  3.94047254,  4.61711977,  6.24903233,  9.55266057, 11.98062847]
theta_cut_ = interp1d(delta_G_mid, theta_cut_vals, kind = 'linear', fill_value = 'extrapolate', assume_sorted = True)

catalog_theta0_vec = theta_cut_(catalog_delta_G_vec)
catalog_data = np.array(np.vstack((catalog_sep_vec, catalog_dist_vec, catalog_delta_G_vec, catalog_R_chance_align_vec, catalog_theta0_vec)))

theta_mat = np.outer( 1/catalog_data[1,:] , mid_data * AU_pc)   # \theta_ij = s_j / d_i
theta0_T = catalog_data[-1,:]    # \theta_0_i
theta0_mat = (np.ones(np.shape(theta_mat)).T * theta0_T).T   # (1 \times \theta_ij)
catalog_f_mat = selection_fct_vec(theta_mat,theta0_mat)    # f(\theta_mat / \theta_0_mat)


# ========================================

# Inputs
p0_bounds = [[0,3],[-3,3],[-3,log_f]]
power_law_index = 1.73
power_law_index_ca = -1

N_data = np.size(catalog_data[0,:])

# --------------------------------------------------

# load initial separation distribution (neglect normalization)
def dn0da_fct(a,l):
    return a**-l 

# evolve separation distribution to T ~ 10 Gyr
def phi(free_params):
    power_index = free_params[0]
    log_f = free_params[-1]
    f = 10**(log_f)

    # evolve histogram to T ~ 10 Gyr (depends on f)
    delta_t_val_f = delta_t_(m,inv_v_avg,f,p_max)    # delta_t value
    N_f = int(T / delta_t_val_f)    # number of time steps
    timestep = N_f - 1

    dn0da = dn0da_fct(mid,power_index)
    delta_a = np.diff(bins)      # bin spacing vector
    dnda = []                       # evolved histogram counts
    for i in range(len(mid)): 
        a0 = results_list[i][0]
        p_i = np.array([results_list[j][1][1][0][i,timestep] for j in range(len(mid))])        # survival probability 
        dnda = np.concatenate(([dnda, [np.sum(p_i * delta_a * dn0da)]]))
    return dnda / np.sum(np.diff(bins) * dnda) # notice we need to normalize since the rv_histogram.pdf is normalized

def phi_ca(free_params):
    power_index_ca = free_params[1]
    return dn0da_fct(mid, power_index_ca)

# # see contamination_rate_fit local file for plot
# def contamination_probability_(s,parameter_vec):
#     polynomial_ = np.poly1d(parameter_vec)
#     polynomial_value = polynomial_(s)
#     return polynomial_value * np.heaviside(polynomial_value,0)

def theta_is_within_bounds(theta, theta_bounds):
    '''
    helper function for flat priors or restricted likelihoods. 
    theta: array of floats, e.g. [1, 2, 3]
    theta_bounds: array of constraints, e.g. [[0, None], [0, 3], [None, 5] 
    If there's a None in theta_bounds, the prior will be improper. 
    '''
    for i, param in enumerate(theta):
        this_min, this_max  = theta_bounds[i]
        
        if this_min is None:
            this_min = -np.inf
        if this_max is None:
            this_max = np.inf
        if not (this_min <= param < this_max):
            return False
    return True

# Contamination rate accounted for. Needed to restrict integration to data separations
# Otherwise, contamination probability would be greater than 1 and give an NAN log(denominator)
def log_likelihood(free_params, inputs):
    power_index = free_params[0]
    power_index_ca = free_params[1]
    log_f = free_params[-1]
    
    p0_bounds = inputs[-1]
    if theta_is_within_bounds(free_params, p0_bounds) == False:
        return -np.inf
    
    catalog_data = inputs[0]
    
    phi_vec = phi(free_params)
    phi_fct = rv_histogram([phi_vec,bins])
    
    numerator = phi_fct.pdf(catalog_data[0,:])
    Phi = np.diff(bins_data) * phi_vec[n_min:n_max] 
    denominator = np.dot(catalog_f_mat,Phi)
    
    prob = (1 - catalog_data[3,:]) * numerator / denominator
    
    phi_ca_vec = phi_ca(free_params)
    phi_ca_fct = rv_histogram([phi_ca_vec,bins])
    
    numerator_ca = phi_ca_fct.pdf(catalog_data[0,:])
    Phi_ca = np.diff(bins_data) * phi_ca_vec[n_min:n_max] 
    denominator_ca = np.dot(catalog_f_mat,Phi_ca)
    
    prob_ca = catalog_data[3,:] * numerator_ca / denominator_ca
    return np.sum(np.log(prob + prob_ca))

# ========================================
# Maximizing the log-likelihood
free_params_0 = np.array([1.618096 ,-1, -1])  # obtained using no perturber broken power law fit (and eyeballing)
p0_bounds = [[0,3],[-3,1],[-3,0.2]]

ti = time.time()

res = minimize(lambda free_params, inputs: -log_likelihood(free_params, inputs),free_params_0,args = [catalog_data,p0_bounds],
                        method = 'Nelder-Mead', 
                        options={'maxiter':2000,'maxfev':2000,'adaptive':True,'xatol':0.0000001,'fatol':0.0000001})

calc_time_minimization = time.time() - ti

print('---')
print('Finished Minimization Procedure: ' + str.format('{0:.2f}',calc_time_minimization/60) +  ' min')
print(res)
print('--- \n')

# ========================================
# Initializing the log posterior

def log_flat_prior(theta, theta_bounds):
    '''
    theta: array of parameters, e.g. [1, 2, 3]
    theta_bounds: array of the same length, but with a list of length
        two (lower and upper bounds) at each element. e.g.
        [[0, 2], [1, 3], [2, 6]]
    '''
    if theta_is_within_bounds(theta, theta_bounds):
        return 0
    else: 
        return -np.inf

def log_posterior(free_params, inputs):
    p0_bounds = inputs[-1]
    lnprior = log_flat_prior(theta = free_params, theta_bounds = p0_bounds)
    if np.isfinite(lnprior):
        lnlikelihood = log_likelihood(free_params,inputs)
    else:
        lnlikelihood = 0 
    return lnprior + lnlikelihood

# ========================================
# Inputs for emcee

nwalkers = int(sys.argv[13])
str_nwalkers = str(nwalkers)

burn = int(sys.argv[14])
str_burn = str(burn)

n_steps = int(sys.argv[15])
str_n_steps = str(n_steps)

# -----------------------------------------
# Running the emcee code

# Set initial walker positions
def get_good_p0_ball(p0, theta_bounds, nwalkers, r = 0.01):
    '''
    Utility function for initializing MCMC walkers. Returns walkers clustered around a 
    point p0 in parameter space, all of which fall within theta_bounds. 
    
    p0: point in parameter space that we think might have a high probability. e.g. [1, 2, 3]
    theta_bounds: the range of parameter space within which we'll allow the walkers to explore;
        we have flat priors within this region. E.g. [[0, 2], [1, 3], [2, 4]]
    nwalkers: int; number of walkers to initialize
    r: float. Decreasing this makes the walkers more and more clustered around p0
    '''
    num_good_p0 = 0

    ball_of_p0 = []
    while num_good_p0 < nwalkers:
        #suggested_p0 = p0 + np.array([r*j*np.random.randn() for j in p0])
        suggested_lambda = p0[0] + r*p0[0]*np.random.randn()
        suggested_lambda_ca = p0[1] + r*p0[1]*np.random.randn()
        suggested_log_fp = p0[-1] + r*np.random.randn()
        suggested_p0 = np.array([suggested_lambda,suggested_lambda_ca,suggested_log_fp])
        suggested_p0_prob = log_flat_prior(suggested_p0, theta_bounds = theta_bounds)
        if np.isfinite(suggested_p0_prob):
            ball_of_p0.append(suggested_p0)
            num_good_p0 += 1
    return ball_of_p0

file_name = ( 'emcee_splaw_' + str_log_m + '_' + str_log_K + '_'
             + bin_scale + '_' + str_log_alow + '_' + str_log_ahigh + '_' + str_step_size + '_'
             + str_log_alow_offset + '_' + str_log_ahigh_offset + '_'
             + str_nwalkers + '_' + str_burn + '_' + str_n_steps )

inputs = [catalog_data,p0_bounds]
p0 = res.x
ndim = len(p0)
theta_bounds = inputs[-1]
p0_ball = get_good_p0_ball(p0 = p0, theta_bounds = theta_bounds, nwalkers = nwalkers, r = 0.01)

calc_time_emcee = [calc_time_minimization]

print('initialized walkers... burning in...')
sampler = emcee.EnsembleSampler(nwalkers = nwalkers, dim = ndim, lnpostfn = log_posterior, args = [inputs], threads = 1)
pos, prob, state = sampler.run_mcmc(p0_ball, burn)
sampler.reset()
print('completed burn in ...')

calc_time_emcee = np.concatenate((calc_time_emcee, [time.time() - ti]))

for i, result in enumerate(sampler.sample(pos, iterations = n_steps)):
    if (i+1) % 50 == 0:    
        print("{0:5.1%}".format(float(i) / n_steps))
        calc_time_emcee = np.concatenate((calc_time_emcee, [time.time() - ti]))
        
# ========================================

# Saving all the results (ignore dimension_master and separation distributions)

emcee_res = [sampler.flatchain, sampler.chain, sampler.lnprobability]
calc_time_emcee = np.concatenate((calc_time_emcee, [time.time() - ti]))
result = [emcee_res,catalog_data,res,calc_time_emcee]
np.save(file_name,result)

print('Successful Run!')
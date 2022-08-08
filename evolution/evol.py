# For previous data, see _old subscripts for code and data files

# load packages and functions
import timeit as timeit
import sys
import os
import numpy as np
import math as math 
from scipy.stats import maxwell
from scipy.integrate import quad, dblquad
from scipy.special import erf, hyp2f1

ti = timeit.default_timer()

## input parameters
a0 = float(sys.argv[1])    # initial binary semimajor axis (pc)    # perturber mass (M_solar)
str_a0 = str.format('{0:.9f}',a0)
log_m = float(sys.argv[2]) # perturber mass (M_solar)
str_log_m = str.format('{0:.3f}',log_m)
m = 10**log_m
log_rs = float(sys.argv[3]) # NFW scale radius (pc)
str_log_rs = str.format('{0:.3f}',log_rs)
rs = 10**log_rs
alpha = float(sys.argv[4]) # power law index
str_alpha = str.format('{0:.3f}',alpha)
if alpha == -3:
    alpha = -2.99999
log_f = float(sys.argv[5]) # halo fraction [setting max number of time steps]
f = 10**log_f
str_log_f = str.format('{0:.3f}',log_f)

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

## other masses
#M = m1 + m2 # total binary mass
#mu = m1*m2 / (m1 + m2) # reduced mass
#k = G * mu * M # U(p) = -k/rk/r

# ------------------------------------------------------

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

sigma = v0/math.sqrt(2)/math.sqrt(2) # divide once due to 2 in denominator, divide again since vc = rel vel dispersion
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

log_eps = float(sys.argv[6])   # threshold fractional energy injection defining p_max
energy_fraction = 10**log_eps
str_log_eps = str.format('{0:.2f}',log_eps)
p_max = p_max_(1,m,1,inv_v_avg,1,energy_fraction) # p_max value (f=1 for efficiency)
delta_t = delta_t_(m,inv_v_avg,f,p_max)    # delta_t value
N = int(T / delta_t)    # number of time steps

# ------------------------------------------------------

# load samples
## parameters
log_K = float(sys.argv[7])  # v0.3: number of binaries fixed
str_log_K = str.format('{0:.2f}',log_K)
K = int(10**log_K)  
N_enc = int(N*K)   # v0.3: number of encounters calculated
R = p_max   # maximum impact parameter (pc)

#log_N_enc_approx = str.format('{0:.2f}',log_N_enc_approx)
#str_log_N_enc = str.format('{0:.2f}',log_N_enc_approx)
#N_enc_approx = int(10**log_N_enc_approx)   # number of samples
#N_enc = int(N_enc_approx / N) * N
#R = p_max   # maximum impact parameter (pc)
#K = int(N_enc / N) # number of binaries

# ------------------------------------------------------
## generate encounter samples

def v_rel_sampler(vc,vesc,N_enc):
    ## relative velocity sampler for truncated MB distributions
    ### note, divide by sqrt(2) because maxwell.rvs automatically has a 2 in denominator of exponential
    ### divide by sqrt(2) one more time since v_c is the MACHO-MACHO relative velocity dispersion 
    ### NOT the velocity dispersion of stars and halos(01/16/2022)
    sigma_v = vc / math.sqrt(2) / math.sqrt(2)
    v1_sample = maxwell.rvs(loc = 0, scale = sigma_v, size = N_enc)
    N_v = len(v1_sample[v1_sample>vesc])
    while N_v > 0:
        v1_sample[v1_sample>vesc] = maxwell.rvs(loc = 0, scale = sigma_v, size = N_v)
        N_v = len(v1_sample[v1_sample>vesc])

    v2_sample = maxwell.rvs(loc = 0, scale = sigma_v, size = N_enc)
    N_v = len(v2_sample[v2_sample>vesc])
    while N_v > 0:
        v2_sample[v2_sample>vesc] = maxwell.rvs(loc = 0, scale = sigma_v, size = N_v)
        N_v = len(v2_sample[v2_sample>vesc])

    phi1_sample = np.random.random(size = N_enc) * 2 * np.pi
    phi2_sample = np.random.random(size = N_enc) * 2 * np.pi
    theta1_sample = np.arccos(2 * ( np.random.random(size = N_enc) ) - np.ones(N_enc))
    theta2_sample = np.arccos(2 * ( np.random.random(size = N_enc) ) - np.ones(N_enc))

    vx1_sample = v1_sample * np.sin(theta1_sample) * np.cos(phi1_sample)
    vx2_sample = v2_sample * np.sin(theta2_sample) * np.cos(phi2_sample)
    vy1_sample = v1_sample * np.sin(theta1_sample) * np.sin(phi1_sample)
    vy2_sample = v2_sample * np.sin(theta2_sample) * np.sin(phi2_sample)
    vz1_sample = v1_sample * np.cos(theta1_sample)
    vz2_sample = v2_sample * np.cos(theta2_sample)

    vx_rel_sample = vx1_sample - vx2_sample
    vy_rel_sample = vy1_sample - vy2_sample
    vz_rel_sample = vz1_sample - vz2_sample

    v_rel_sample = np.sqrt(vx_rel_sample**2 + vy_rel_sample**2 + vz_rel_sample**2)
    return v_rel_sample
    
p_sample = np.sqrt(np.random.random(size = N_enc)) * R     # PDF: 2p
theta_sample = np.arccos(2 * ( np.random.random(size = N_enc) ) - np.ones(N_enc))   # PDF: sin \theta / 2
phi_sample = np.random.random(size = N_enc) * 2 * np.pi    # PDF: Uniform [0,2\pi)
gamma_sample = np.random.random(size = N_enc) * 2 * np.pi    # PDF: Uniform [0,2\pi)
v_sample = v_rel_sampler(v0,vesc)    # PDF: Maxwellian with velocity dispersion v0 [0,\infty)
tau_rand = np.random.random(size = N_enc) # PDF for effective observation time

# ------------------------------------------------------
# Load tau(psi,e) matrix

local_dir = '/het/p4/ramirez/dmbinaries/shooter_edr3/newgen_pipeline'

## find tau corresponding to eccentric anomaly and eccentricity
def find_nearest_(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    
Nres = 10000
psi_discrete = np.linspace(0,2*np.pi,Nres)
e_discrete = np.linspace(0,1,Nres)
tau_mat = np.load(os.path.join(local_dir, 'tau_mat_' + str(np.log10(Nres)) + '.npy'), allow_pickle = True)

# ------------------------------------------------------
# load and evolve monochromatic binary population

## reshape encounters to distribute encounters evenly over all K binaries
p_sample=p_sample.reshape(-1,N)
theta_sample=theta_sample.reshape(-1,N)
phi_sample=phi_sample.reshape(-1,N)
v_sample=v_sample.reshape(-1,N)
gamma_sample=gamma_sample.reshape(-1,N)
tau_rand=tau_rand.reshape(-1,N)

## load functions relevant to binary evolution
# Analytical Definition of U(p) (Subject to calculational breaking down)
# See Appendix of Report for neater formulas

## binary period
def period_(a,k):
    return 2*np.pi*a**(3/2)*np.sqrt(1/k) * ( (km_m) / (pc_m) ) * ( (Gyr_yr) * (yr_s) ) # Gyr

def Ua_(p,R,alpha):
    if p < R:
        x = R*R/p/p
        x_inv = 1/x
        return ( 1 - (math.sqrt(-x_inv + 1)) + 
                (math.sqrt(-1 + x)*hyp2f1(0.5,(3 + alpha)/2.,1.5,1 - x_inv))/(x*x*(R/p)**alpha) )
    else:
        return 1

# r calculation needs to be done separately because the find_nearest_ command cannot be 
# vectorized; so it must be loaded

def r_(tau,a,e):
    if a == np.inf:
        return [np.inf,1]
    tau = np.modf(tau)[0]
    n_e_eff = find_nearest_(e_discrete,e)
    e_eff = e_discrete[n_e_eff]
    psi = psi_discrete[find_nearest_(tau_mat[:,n_e_eff], tau)]
    return [a * (1 - e * math.cos(psi)),psi]

# need to calculate these using a for loop to produce r_sample

def orbital_parameters_(ai,ei,r,psi,k,p,theta,phi,gamma,vp,Mp,rs,alpha):
    if ai == np.inf or ei >= 1:
        return np.array([np.inf,ei,-1])
    
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    cos_2_theta = math.cos(2*theta)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_gamma = math.sin(gamma)
    cos_gamma = math.cos(gamma)
    
    # velocity calculation 
    li = np.sqrt(k * ai * (1.-ei*ei) )
    eta_dot = li / r / r

    Ei = k/2/ai
    if psi <= np.pi:
        r_dot = math.sqrt(np.abs(2 * (-Ei + k/r - r*r*eta_dot*eta_dot/2)))
    else:
        r_dot = -math.sqrt(np.abs(2 * (-Ei + k/r - r*r*eta_dot*eta_dot/2)))
        
    v_x = r_dot * sin_theta * cos_phi + r * eta_dot * (- cos_theta * cos_phi * sin_gamma - sin_phi * cos_gamma)
    v_y = r_dot * sin_theta * sin_phi + r * eta_dot * (- cos_theta * sin_phi * sin_gamma + cos_phi * cos_gamma)
    v_z = r_dot * cos_theta + r * eta_dot * sin_gamma*sin_theta
    
    # delta_E calculation
    delta_v_factor = 2*G*Mp/vp
    
    a = r/2
    rho1_sq = p*p+a*a*(1-cos_2_theta)/2+2*a*p*sin_theta*cos_phi
    rho2_sq = p*p+a*a*(1-cos_2_theta)/2-2*a*p*sin_theta*cos_phi
    U1 = Ua_(math.sqrt(rho1_sq),rs,alpha)
    U2 = Ua_(math.sqrt(rho2_sq),rs,alpha)
    U1_rho1_sq = U1/rho1_sq
    U2_rho2_sq = U2/rho2_sq
    delta_v_eff_sq = U1*U1_rho1_sq+U2*U2_rho2_sq-U1_rho1_sq*U2_rho2_sq*(2*p*p-a*a*(1-cos_2_theta))
    
    r2_proj_xy = a * sin_theta
    r2_proj_x = r2_proj_xy * cos_phi
    r2_proj_y = r2_proj_xy * sin_phi
    
    delta_v_eff_x = - U1_rho1_sq * (p + r2_proj_x) + U2_rho2_sq * (p - r2_proj_x)
    delta_v_eff_y = - U1_rho1_sq * r2_proj_y - U2_rho2_sq * r2_proj_y
    v_dot_delta_v_eff = v_x * delta_v_eff_x + v_y * delta_v_eff_y
    
    delta_E = delta_v_factor * ( delta_v_factor * delta_v_eff_sq / 2 + v_dot_delta_v_eff )

    # delta_l calculation
    delta_l_x = delta_v_factor * r * r2_proj_y * cos_theta * (U1_rho1_sq + U2_rho2_sq)
    delta_l_y = delta_v_factor * r * cos_theta * (p * (U2_rho2_sq - U1_rho1_sq) - r2_proj_x * (U1_rho1_sq + U2_rho2_sq) )
    delta_l_z = delta_v_factor * 2 * p * r2_proj_y * (U1_rho1_sq - U2_rho2_sq)
    
    # semimajor axis calculation
    Ef = Ei-delta_E
    if Ef>0:
        af = k/2/Ef
    else:
        return np.array([np.inf,ei,-1])
    
    # eccentricity calculation
    li_x = li * (-cos_theta * cos_phi * cos_gamma + sin_phi * sin_gamma)
    li_y = li * (-cos_theta * sin_phi * cos_gamma - cos_phi * sin_gamma)
    li_z = li * cos_gamma * sin_theta
    lf = math.sqrt((delta_l_x + li_x)*(delta_l_x + li_x) + (delta_l_y + li_y)*(delta_l_y + li_y) +
                 (delta_l_z + li_z)*(delta_l_z + li_z))
    ef = math.sqrt(1-lf*lf/k/af)
    if ef >= 1:
        return np.array([np.inf,ei,-1])
    
    # new initial time calculation
    vf_x = v_x + delta_v_factor * delta_v_eff_x
    vf_y = v_y + delta_v_factor * delta_v_eff_y
    vf_z = v_z
    vf_r = vf_x * sin_theta * cos_phi + vf_y * sin_theta * sin_phi + vf_z * cos_theta
    cos_psif = (1 - (1 - ei * math.cos(psi)) * ai/af) / ef
    if abs(cos_psif) > 1:
        cos_psif = cos_psif / abs(cos_psif)
    if vf_r >= 0:
        psif = math.acos(cos_psif)
    else:
        psif = - math.acos(cos_psif)  + 2 * math.pi
    tauf = (psif - ef * math.sin(psif)) / 2 / math.pi
    
    return np.array([af,ef,tauf],dtype=np.float64)

## bin construction
bin_scale = sys.argv[8]

log_alow = float(sys.argv[9])
str_log_alow = str.format('{0:.3f}',log_alow)
log_ahigh = float(sys.argv[10])
str_log_ahigh = str.format('{0:.3f}',log_ahigh)
step_size = float(sys.argv[11]) # number of bins
str_step_size = str(step_size)
#str_step_size = str.format('{0:.3f}',step_size)

log_alow_offset = float(sys.argv[12])
str_log_alow_offset = str.format('{0:.3f}',log_alow_offset)
log_ahigh_offset = float(sys.argv[13])
str_log_ahigh_offset = str.format('{0:.3f}',log_ahigh_offset)

## preload evolved projected separation array
#s_sample = np.zeros((p_sample.shape[0],p_sample.shape[1]+1))

# make the array of semimajor axis values
if bin_scale=='log': 
    #log_bins_hist = np.arange(log_alow,log_ahigh,step_size)
    #bins_hist = 10**log_bins_hist
    
    bins_finite = np.arange(log_alow,log_ahigh+step_size,step_size)  # bins
    bins_finite = 10**bins_finite
    bins_hist = np.arange(log_alow+log_alow_offset,log_ahigh+log_ahigh_offset+step_size,step_size)
    bins_hist = 10**bins_hist

if bin_scale=='lin':
    alow_offset=10**(log_alow)
    ahigh_offset=10**(log_ahigh)
    
    #alow=10**log_alow
    #ahigh=10**log_ahigh
    
    Nbins = 180
    bins_finite = np.linspace(alow_offset,ahigh_offset,Nbins)
    #bins_hist = np.linspace(alow,ahigh,Nbins)  # bins

# generate initial binary semimajor axis values
### a_init = a0*np.ones(p_sample.shape[0]) # initial semimajor axis values
log_bins_finite = np.arange(log_alow,log_ahigh+step_size,step_size)
n_a0_low = np.where(log_bins_finite <= np.log10(a0))[0][-1]

log_bin_min = log_bins_finite[n_a0_low]
log_bin_max = log_bins_finite[n_a0_low + 1]
log_a_init = log_bin_min * np.ones(p_sample.shape[0]) + (log_bin_max - log_bin_min) * np.random.random(p_sample.shape[0])# initial semimajor axis values

# generate other initial orbital parameters
a_sample = 10**log_a_init
e_sample = np.sqrt(np.random.random(p_sample.shape[0]))
#q_sample = np.load('q_data.npy', allow_pickle = True) [Evolution independent of q]
M_data = np.load(os.path.join(local_dir, 'M_data' + '.npy'), allow_pickle = True)
rand_ind = np.random.randint(0, high=4350, size=K, dtype=int)
M_sample = np.array([M_data[n] for n in rand_ind])
#M_sample = np.ones(K)
#mu_sample = q / (1 + q)**2 * M
k_sample = G * M_sample # U(p) = -k/r

# generate orientations of binaries relative to line of sight 
## we generate these only once since different timesteps will correspond to different perturber fractions "f"
i_sample = np.arccos(2 * ( np.random.random(size = K) ) - np.ones(K))

## evolve binaries
### initial timestep [performed to columns of histogram counts and errors]
timestep = 0
tau_sample_obs = tau_rand[:,timestep]

### Obtain physical separation given orbital parameters
r_psi_mat = np.array([r_(tau_sample_obs[n],a_sample[n],e_sample[n]) for n in range(K)])
r_sample = r_psi_mat[:,0]
psi_sample = r_psi_mat[:,1]

### Calculate the histogram of projected separations
s_sample = r_sample * np.sin(i_sample)

n, bins = np.histogram(s_sample, 
                      np.concatenate((bins_hist, [np.PINF])), range = [0,1], density = False) # histogram of binary seps
p_mat = n / np.diff(bins) / K  # density=True
p_err_mat = np.sqrt(n) / np.diff(bins) / K #error in density estimation (assume large counts)
binary_loss_fraction_list = [ n[-1] / K ]
binary_loss_fraction_err_list = [ np.sqrt(n[-1]) / K ] 

### First encounter
orbital_parameters = np.array([orbital_parameters_(a_sample[n],e_sample[n],r_sample[n],psi_sample[n],
                                                   k_sample[n],
                                                   p_sample[n,timestep],theta_sample[n,timestep],
                                                   phi_sample[n,timestep],gamma_sample[n,timestep],v_sample[n,timestep],m,rs,alpha) 
                               for n in range(K)])
a_sample = orbital_parameters[:,0]
e_sample = orbital_parameters[:,1]
tau_sample = orbital_parameters[:,2]  

for timestep in range(1,p_sample.shape[1]):
    ### observation time occurs at random time in between encounter
    tau_sample_obs = tau_sample + tau_rand[:,timestep] * delta_t / period_(a_sample, k_sample)
    
    ### Obtain physical separation given orbital parameters
    r_psi_mat = np.array([r_(tau_sample_obs[n],a_sample[n],e_sample[n]) for n in range(K)])
    r_sample = r_psi_mat[:,0]
    
    ### Calculate the histogram of projected separations
    s_sample = r_sample * np.sin(i_sample)
    
    ### save histogram
    n, bins = np.histogram(s_sample, 
                          np.concatenate((bins_hist, [np.PINF])), range = [0,1], density = False)
    p = n / np.diff(bins) / K
    p_err = np.sqrt(n) / np.diff(bins) / K
    
    p_mat = np.column_stack([p_mat, p])
    p_err_mat = np.column_stack([p_err_mat, p_err])
    binary_loss_fraction_list.append(n[-1] / K)
    binary_loss_fraction_err_list.append(np.sqrt(n[-1]) / K)
    
    #### evolve to next timestep
    tau_sample = tau_sample + delta_t / period_(a_sample, k_sample)

    ### Obtain physical separation given orbital parameters
    r_psi_mat = np.array([r_(tau_sample[n],a_sample[n],e_sample[n]) for n in range(K)])
    r_sample = r_psi_mat[:,0]
    psi_sample = r_psi_mat[:,1]
    
    #### Perturber effect on binary
    orbital_parameters = np.array([orbital_parameters_(a_sample[n],e_sample[n],r_sample[n],psi_sample[n],
                                                       k_sample[n],
                                                       p_sample[n,timestep],theta_sample[n,timestep],
                                                   phi_sample[n,timestep],gamma_sample[n,timestep],v_sample[n,timestep],m,rs,alpha) 
                               for n in range(K)])
    a_sample = orbital_parameters[:,0]
    e_sample = orbital_parameters[:,1]
    tau_sample = orbital_parameters[:,2]                    
    
results_hist = [p_mat,bins,p_err_mat,
                binary_loss_fraction_list,binary_loss_fraction_err_list]
input_vec = [a0,log_m,log_rs,alpha,log_K]
#input_vec = [a0,log_m,log_rs,alpha,log_N_enc_approx]
fixed_parameter_vec = [f,v0,T]
numbers_vec = [N, K, N_enc]
results = [input_vec, results_hist, fixed_parameter_vec,numbers_vec]

filename = 'hists_' + str_a0
np.save(filename, results)

tf = timeit.default_timer()
print('Run Time (min): ' + str( (tf - ti) / 60 ) )
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Utilities as u
import random
import time
from scipy.stats import norm, shapiro, probplot
from scipy.stats import t as t_dist
import pingouin as pg
from scipy.optimize import fsolve
from scipy.integrate import quad

# Fix the seeds
seed_m = random.randint(1, 10^7)
seed = 10   # Seed for Montecarlo simulation

# Read data from Excel
data = pd.read_excel("CreditModelRisk_RawData.xlsx")

# Point 1
# Part a
# Shapiro-Wilk test univariate
print("-------------------- Point 1 --------------------\n")

# Confidence level
alpha = 1 - 0.999
LGD = 1 - data['RR']

# We consider the inverse of the standard Gaussian in order to verify
# gaussianity (Default points)
k_SG = norm.ppf(data['DR_SG'])  # Speculative grade corporate
k_AR = norm.ppf(data['DR_All_Rated'])  # All grade corporate

# Shapiro tests
W_SG, pValue_SG = shapiro(k_SG)
W_AR, pValue_AR = shapiro(k_AR)
W_LGD, pValue_LGD = shapiro(LGD)

H_SG, pValue_SG_LGD, T = pg.multivariate_normality(pd.concat((data['DR_SG'], LGD), axis = 1))
H_AR, pValue_AR_LGD, T = pg.multivariate_normality(pd.concat((data['DR_All_Rated'], LGD), axis = 1))

print(f"Statistical analysis (alpha = {alpha:.3f})\n")
print(f"Shapiro-Wilk test - Univariate case")
print(f"K_SG p-value: {pValue_SG:.4f}")
print(f"K_AR p-value: {pValue_AR:.4f}")
print(f"Loss given default p-value: {pValue_LGD:.4f}\n")
print("Henke-Zikler test test - Bivariate case")
print(f"K_SG - LGD p-value: {pValue_SG_LGD:.4f}")
print(f"K_SG - LGD p-value: {pValue_AR_LGD:.4f}")

# Pearson coefficients
corr_SG = np.corrcoef(k_SG, LGD)[0, 1]
corr_AR = np.corrcoef(k_AR, LGD)[0, 1]

# Pearson coefficients confidence level
alpha = 1 - 0.95

# z transformation values
z_SG = np.log((1 + corr_SG) / (1 - corr_SG)) / 2
z_AR = np.log((1 + corr_AR) / (1 - corr_AR)) / 2

# size interval
n = len(data['DR_SG'])

# z transformation intervals
z_SG_interval = [z_SG - norm.ppf(1 - alpha / 2) * np.sqrt(1 / (n - 3)),
                 z_SG + norm.ppf(1 - alpha / 2) * np.sqrt(1 / (n - 3))]
z_AR_interval = [z_AR - norm.ppf(1 - alpha / 2) * np.sqrt(1 / (n - 3)),
                 z_AR + norm.ppf(1 - alpha / 2) * np.sqrt(1 / (n - 3))]

# Pearson coefficient intervals
corr_SG_interval = [(np.exp(2 * z_SG_interval[0]) - 1) / (np.exp(2 * z_SG_interval[0]) + 1),
                    (np.exp(2 * z_SG_interval[1]) - 1) / (np.exp(2 * z_SG_interval[1]) + 1)]
corr_AR_interval = [(np.exp(2 * z_AR_interval[0]) - 1) / (np.exp(2 * z_AR_interval[0]) + 1),
                    (np.exp(2 * z_AR_interval[1]) - 1) / (np.exp(2 * z_AR_interval[1]) + 1)]

print(f"Pearson coefficient kAR - LGD: {corr_AR:.4f} - CI = [{corr_AR_interval[0]:.4f}, {corr_AR_interval[1]:.4f}]")
print(f"Pearson coefficient kSG - LGD: {corr_SG:.4f} - CI = [{corr_SG_interval[0]:.4f}, {corr_SG_interval[1]:.4f}]")

# Linear regression coefficients
a_AR = np.polyfit(k_AR, LGD, 1)
a_SG = np.polyfit(k_SG, LGD, 1)

# Linear regression values
k_AR_reg = np.polyval(a_AR, k_AR)
k_SG_reg = np.polyval(a_SG, k_SG)

# Plot - Correlation between k and LGD
plt.figure()
plt.plot(k_AR, LGD, 'ro', label='K_AR - LGD')
plt.plot(k_AR, k_AR_reg, 'blue', label='Linear Regression')
plt.title('Scatter plot K_AR - LGD with linear regression')
plt.xlabel('K_AR')
plt.ylabel('LGD')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(k_SG, LGD, 'ro', label='K_SG - LGD')
plt.plot(k_SG, k_SG_reg, 'blue', label='Linear Regression')
plt.title('Scatter plot K_SG - LGD with linear regression')
plt.xlabel('K_SG')
plt.ylabel('LGD')
plt.legend()
plt.grid(True)

plt.show()

# Point 2
print("\n--------------------- Point 2 --------------------\n\n")

# Point a
# Calcola i valori medi per l'approccio naive
LGD_hat = np.mean(1 - data['RR'])  # Media Loss Given Default
PD_AR_hat = np.mean(data['DR_All_Rated'])  # Media probabilità di default per tutte le valutazioni
PD_SG_hat = np.mean(data['DR_SG'])  # Media probabilità di default per la valutazione dei gradi speculativi

# Livelli di confidenza
alpha = np.array([0.999, 0.99])

# Naive approach
# Expected Loss
EL_naive_SG = LGD_hat * PD_SG_hat  # Speculative grade
EL_naive_AR = LGD_hat * PD_AR_hat  # All rated grade

# Regulatory capital in naive approach
# Speculative grade
RC_naive_SG = LGD_hat * norm.cdf((norm.ppf(PD_SG_hat) - np.sqrt(u.correlation_IRB(PD_SG_hat)) *
                                   norm.ppf(1 - alpha)) / np.sqrt(1 - u.correlation_IRB(PD_SG_hat))) - EL_naive_SG

# All rated grade
RC_naive_AR = LGD_hat * norm.cdf((norm.ppf(PD_AR_hat) - np.sqrt(u.correlation_IRB(PD_AR_hat)) *
                                   norm.ppf(1 - alpha)) / np.sqrt(1 - u.correlation_IRB(PD_AR_hat))) - EL_naive_AR

# General approach
# Number of simulations
N = int(3*1e6)

# Speculative k standard deviation
sigma_SG = np.std(k_SG)

# All rated k standard deviation
sigma_AR = np.std(k_AR)

# Recovery rate standard deviation
sigma_LGD = np.std(1 - data['RR'])

# k hat (k mean obtained by inverting PD) computation
def f_SG(k):
    integrand = lambda x: np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi) * norm.cdf(k + sigma_SG * x)
    integral_result, _ = quad(integrand, -np.inf, +np.inf)
    return integral_result - PD_SG_hat

k_hat_SG = fsolve(f_SG, np.mean(k_SG))

def f_AR(k):
    integrand = lambda x: np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi) * norm.cdf(k + sigma_AR * x)
    integral_result, _ = quad(integrand, -np.inf, +np.inf)
    return integral_result - PD_AR_hat

k_hat_AR = fsolve(f_AR, np.mean(k_AR))

# QQ-plot and histogram for k_AR
plt.figure()
plt.subplot(1, 2, 1)
probplot(k_AR, dist="norm", plot=plt)
plt.title('QQ-plot k_AR')
plt.subplot(1, 2, 2)
plt.hist(k_AR, bins='auto', density=True)
plt.title('Density - Histogram k_AR')
plt.xlabel('k_AR')
plt.ylabel('Density')
plt.plot(np.linspace(min(k_AR), max(k_AR), 100),
         (1 / (sigma_AR * np.sqrt(2 * np.pi))) *
         np.exp(-0.5 * ((np.linspace(min(k_AR) - 0.1, max(k_AR) + 0.1, 100) - k_hat_AR) / sigma_AR) ** 2))

# QQ-plot and histogram for k_SG
plt.figure()
plt.subplot(1, 2, 1)
probplot(k_SG, dist="norm", plot=plt)
plt.title('QQ-plot k_S_G')
plt.subplot(1, 2, 2)
plt.hist(k_SG, bins='auto', density=True)
plt.title('Density - Histogram k_S_G')
plt.xlabel('k_S_G')
plt.ylabel('Density')
plt.plot(np.linspace(min(k_SG), max(k_SG), 100),
         (1 / (sigma_SG * np.sqrt(2 * np.pi))) *
         np.exp(-0.5 * ((np.linspace(min(k_SG) - 0.1, max(k_SG) + 0.1, 100) - k_hat_SG) / sigma_SG) ** 2))

# QQ-plot and histogram for LGD
plt.figure()
plt.subplot(1, 2, 1)
probplot(LGD, dist="norm", plot=plt)
plt.title('QQ-plot LGD')
plt.subplot(1, 2, 2)
plt.hist(LGD, bins='auto', density=True)
plt.title('Density - Histogram LGD')
plt.xlabel('LGD')
plt.ylabel('Density')
plt.plot(np.linspace(min(LGD), max(LGD), 100),
         (1 / (sigma_LGD * np.sqrt(2 * np.pi))) *
         np.exp(-0.5 * ((np.linspace(min(LGD) - 0.1, max(LGD) + 0.1, 100) - LGD_hat) / sigma_LGD) ** 2))

plt.show()

# QQ-plots and histograms
# Market parameters simulation
np.random.seed(seed_m)
M = np.random.randn(N)

# k and LGD terms simulation
np.random.seed(seed)
g = np.random.randn(2, N)  # LGD and K standard gaussian simulation

### LHP portfolio ####

# Montecarlo simulation
print("\n------------- Montecarlo Regulatory capital LHP Stress - Add on computation ----------------\n")

# LGD simulation, fixing K
start_time = time.time()
EL_vec_SG_LGD, EL_SG_LGD, RC_SG_LGD, RC_SG_LGD_CI, EL_SG_LGD_CI = u.montecarlo_RC(N, norm.ppf(PD_SG_hat), sigma_SG, LGD_hat, sigma_LGD, alpha, g, M, 1, corr_SG)
EL_vec_AR_LGD, EL_AR_LGD, RC_AR_LGD, RC_AR_LGD_CI, EL_AR_LGD_CI = u.montecarlo_RC(N, norm.ppf(PD_AR_hat), sigma_AR, LGD_hat, sigma_LGD, alpha, g, M, 1, corr_AR)
print("LGD simulation - Montecarlo simulations computation time: {:.2f} seconds".format(time.time() - start_time))

# Add-on computation
Add_on_SG_LGD, Add_on_SG_LGD_CI = u.Add_on(RC_SG_LGD, RC_naive_SG, EL_SG_LGD, EL_naive_SG, RC_SG_LGD_CI, EL_SG_LGD_CI)
Add_on_AR_LGD, Add_on_AR_LGD_CI = u.Add_on(RC_AR_LGD, RC_naive_AR, EL_AR_LGD, EL_naive_AR, RC_AR_LGD_CI, EL_AR_LGD_CI)

# K simulation, fixing LGD
start_time = time.time()
EL_vec_SG_K, EL_SG_K, RC_SG_K, RC_SG_K_CI, EL_SG_K_CI = u.montecarlo_RC(N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha, g, M, 2, corr_SG)
EL_vec_AR_K, EL_AR_K, RC_AR_K, RC_AR_K_CI, EL_AR_K_CI = u.montecarlo_RC(N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha, g, M, 2, corr_AR)
print("K simulation - Montecarlo simulations computation time: {:.2f} seconds".format(time.time() - start_time))

# Add-on computation
Add_on_SG_K, Add_on_SG_K_CI = u.Add_on(RC_SG_K, RC_naive_SG, EL_SG_K, EL_naive_SG, RC_SG_K_CI, EL_SG_K_CI)
Add_on_AR_K, Add_on_AR_K_CI = u.Add_on(RC_AR_K, RC_naive_AR, EL_AR_K, EL_naive_AR, RC_AR_K_CI, EL_AR_K_CI)

# K and LGD independent
start_time = time.time()
EL_vec_ind_SG, EL_ind_SG, RC_ind_SG, RC_ind_SG_CI, EL_SG_ind_CI = u.montecarlo_RC(N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha, g, M, 3, corr_SG)
EL_vec_ind_AR, EL_ind_AR, RC_ind_AR, RC_ind_AR_CI, EL_AR_ind_CI = u.montecarlo_RC(N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha, g, M, 3, corr_AR)
print("K and LGD independent - Montecarlo simulations computation time: {:.2f} seconds".format(time.time() - start_time))

# Add-on computation
Add_on_ind_SG, Add_on_ind_SG_CI = u.Add_on(RC_ind_SG, RC_naive_SG, EL_ind_SG, EL_naive_SG, RC_ind_SG_CI, EL_SG_ind_CI)
Add_on_ind_AR, Add_on_ind_AR_CI = u.Add_on(RC_ind_AR, RC_naive_AR, EL_ind_AR, EL_naive_AR, RC_ind_AR_CI, EL_AR_ind_CI)

# K and LGD dependent
start_time = time.time()
EL_vec_dep_SG, EL_dep_SG, RC_dep_SG, RC_dep_SG_CI, EL_SG_dep_CI = u.montecarlo_RC(N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha, g, M, 4, corr_SG)
EL_vec_dep_AR, EL_dep_AR, RC_dep_AR, RC_dep_AR_CI, EL_AR_dep_CI = u.montecarlo_RC(N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha, g, M, 4, corr_AR)
print("K and LGD dependent - Montecarlo simulations computation time: {:.2f} seconds".format(time.time() - start_time))

# Add-on computation
Add_on_dep_SG, Add_on_dep_SG_CI = u.Add_on(RC_dep_SG, RC_naive_SG, EL_dep_SG, EL_naive_SG, RC_dep_SG_CI, EL_SG_dep_CI)
Add_on_dep_AR, Add_on_dep_AR_CI = u.Add_on(RC_dep_AR, RC_naive_AR, EL_dep_AR, EL_naive_AR, RC_dep_AR_CI, EL_AR_dep_CI)

for i in range(len(alpha)):
    print("\n")
    print(f"Add-on (alpha = {alpha[i]:.3f})\n")
    print(f"Only LGD simulation:              SG = {Add_on_SG_LGD[i]:.4f} - IC = [{Add_on_SG_LGD_CI[i, 0]:.4f},{Add_on_SG_LGD_CI[i, 1]:.4f}]      AR = {Add_on_AR_LGD[i]:.4f} - IC = [{Add_on_AR_LGD_CI[i, 0]:.4f},{Add_on_AR_LGD_CI[i, 1]:.4f}]")
    print(f"Only K simulation:                SG = {Add_on_SG_K[i]:.4f} - IC = [{Add_on_SG_K_CI[i, 0]:.4f},{Add_on_SG_K_CI[i, 1]:.4f}]        AR = {Add_on_AR_K[i]:.4f} - IC = [{Add_on_AR_K_CI[i, 0]:.4f},{Add_on_AR_K_CI[i, 1]:.4f}]")
    print(f"LGD - K independent simulation:   SG = {Add_on_ind_SG[i]:.4f} - IC = [{Add_on_ind_SG_CI[i, 0]:.4f},{Add_on_ind_SG_CI[i, 1]:.4f}]      AR = {Add_on_ind_AR[i]:.4f} - IC = [{Add_on_ind_AR_CI[i, 0]:.4f},{Add_on_ind_AR_CI[i, 1]:.4f}]")
    print(f"LGD - K dependent simulation:     SG = {Add_on_dep_SG[i]:.4f} - IC = [{Add_on_dep_SG_CI[i, 0]:.4f},{Add_on_dep_SG_CI[i, 1]:.4f}]      AR = {Add_on_dep_AR[i]:.4f} - IC = [{Add_on_dep_AR_CI[i, 0]:.4f},{Add_on_dep_AR_CI[i, 1]:.4f}]")

# Homogeneous portfolio with various numbers of obligors
N_obligors = np.array([50, 100, 250, 500, 1000])  # Number of obligors
N1 = int(2*1e5)

# Market parameters simulation
np.random.seed(seed_m)
M1 = np.random.randn(N1)

# k and LGD standard normal simulation
np.random.seed(seed)  # Seed for reproducibility
g1 = np.random.randn(2, N1)  # Standard normal distribution for k and LGD

print(
    "\n------------------ Montecarlo Regulatory capital HP Stress - Add on computation -----------------\n\n")

EL_SG_LGD_HP = np.zeros((len(N_obligors), len(alpha)))
RC_SG_LGD_HP = np.zeros((len(N_obligors), len(alpha)))
RC_SG_LGD_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
EL_SG_LGD_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))

EL_AR_LGD_HP = np.zeros((len(N_obligors), len(alpha)))
RC_AR_LGD_HP = np.zeros((len(N_obligors), len(alpha)))
RC_AR_LGD_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
EL_AR_LGD_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))

EL_SG_K_HP = np.zeros((len(N_obligors), len(alpha)))
RC_SG_K_HP = np.zeros((len(N_obligors), len(alpha)))
RC_SG_K_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
EL_SG_K_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))

EL_AR_K_HP = np.zeros((len(N_obligors), len(alpha)))
RC_AR_K_HP = np.zeros((len(N_obligors), len(alpha)))
RC_AR_K_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
EL_AR_K_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))

EL_ind_SG_HP = np.zeros((len(N_obligors), len(alpha)))
RC_ind_SG_HP = np.zeros((len(N_obligors), len(alpha)))
RC_ind_SG_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
EL_SG_ind_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))

EL_ind_AR_HP = np.zeros((len(N_obligors), len(alpha)))
RC_ind_AR_HP = np.zeros((len(N_obligors), len(alpha)))
RC_ind_AR_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
EL_AR_ind_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))

EL_dep_SG_HP = np.zeros((len(N_obligors), len(alpha)))
RC_dep_SG_HP = np.zeros((len(N_obligors), len(alpha)))
RC_dep_SG_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
EL_SG_dep_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))

EL_dep_AR_HP = np.zeros((len(N_obligors), len(alpha)))
RC_dep_AR_HP = np.zeros((len(N_obligors), len(alpha)))
RC_dep_AR_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
EL_AR_dep_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))

Add_on_SG_LGD_HP = np.zeros((len(N_obligors), len(alpha)))
Add_on_SG_LGD_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
Add_on_AR_LGD_HP = np.zeros((len(N_obligors), len(alpha)))
Add_on_AR_LGD_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
Add_on_SG_K_HP = np.zeros((len(N_obligors), len(alpha)))
Add_on_SG_K_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
Add_on_AR_K_HP = np.zeros((len(N_obligors), len(alpha)))
Add_on_AR_K_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
Add_on_ind_SG_HP = np.zeros((len(N_obligors), len(alpha)))
Add_on_ind_SG_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
Add_on_ind_AR_HP = np.zeros((len(N_obligors), len(alpha)))
Add_on_ind_AR_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
Add_on_dep_SG_HP = np.zeros((len(N_obligors), len(alpha)))
Add_on_dep_SG_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))
Add_on_dep_AR_HP = np.zeros((len(N_obligors), len(alpha)))
Add_on_dep_AR_CI_HP = np.zeros((len(N_obligors), len(alpha), 2))

for j in range(len(N_obligors)):
    tic = time.time()
    np.random.seed(seed)

    # Idiosyncratic terms simulation
    epsilon = np.random.randn(N_obligors[j], N1)

    # Fix K, only LGD simulation
    # Montecarlo
    EL_vec_SG_LGD_HP, EL_SG_LGD_HP[j, :], RC_SG_LGD_HP[j, :], RC_SG_LGD_CI_HP[j, :, :], EL_SG_LGD_CI_HP[j, :,:] =(u.montecarlo_RC_HP(N1,
                                                                                 norm.ppf(PD_SG_hat),sigma_SG,LGD_hat,
                                                                                 sigma_LGD,alpha, g1,M1, epsilon,1,
                                                                                 N_obligors[j],corr_SG))
    EL_vec_AR_LGD_HP, EL_AR_LGD_HP[j, :], RC_AR_LGD_HP[j, :], RC_AR_LGD_CI_HP[j, :, :], EL_AR_LGD_CI_HP[j, :,:] = u.montecarlo_RC_HP(N1,
                                                                                 norm.ppf(PD_AR_hat),sigma_AR,LGD_hat,
                                                                                 sigma_LGD,alpha, g1, M1,epsilon,
                                                                                 1,N_obligors[j],corr_AR)

    # Add-on
    Add_on_SG_LGD_HP[j, :], Add_on_SG_LGD_CI_HP[j, :, :] = u.Add_on(RC_SG_LGD_HP[j, :], RC_naive_SG, EL_SG_LGD_HP[j, :],
                                                                  EL_naive_SG, RC_SG_LGD_CI_HP[j, :, :],
                                                                  EL_SG_LGD_CI_HP[j, :, :])
    Add_on_AR_LGD_HP[j, :], Add_on_AR_LGD_CI_HP[j, :, :] = u.Add_on(RC_AR_LGD_HP[j, :], RC_naive_AR, EL_AR_LGD_HP[j, :],
                                                                  EL_naive_AR, RC_AR_LGD_CI_HP[j, :, :],
                                                                  EL_AR_LGD_CI_HP[j, :, :])

    # Fix LGD, only K simulation
    # Montecarlo
    EL_vec_SG_K_HP, EL_SG_K_HP[j, :], RC_SG_K_HP[j, :], RC_SG_K_CI_HP[j, :, :], EL_SG_K_CI_HP[j, :,
                                                                                :] = u.montecarlo_RC_HP(N1, k_hat_SG,
                                                                                                      sigma_SG, LGD_hat,
                                                                                                      sigma_LGD, alpha,
                                                                                                      g1, M1, epsilon,
                                                                                                      2, N_obligors[j],corr_SG)
    EL_vec_AR_K_HP, EL_AR_K_HP[j, :], RC_AR_K_HP[j, :], RC_AR_K_CI_HP[j, :, :], EL_AR_K_CI_HP[j, :,
                                                                                :] = u.montecarlo_RC_HP(N1, k_hat_AR,
                                                                                                      sigma_AR, LGD_hat,
                                                                                                      sigma_LGD, alpha,
                                                                                                      g1, M1, epsilon,
                                                                                                      2, N_obligors[j],corr_AR)

    # Add-on
    Add_on_SG_K_HP[j, :], Add_on_SG_K_CI_HP[j, :, :] = u.Add_on(RC_SG_K_HP[j, :], RC_naive_SG, EL_SG_K_HP[j, :],
                                                              EL_naive_SG, RC_SG_K_CI_HP[j, :, :],
                                                              EL_SG_K_CI_HP[j, :, :])
    Add_on_AR_K_HP[j, :], Add_on_AR_K_CI_HP[j, :, :] = u.Add_on(RC_AR_K_HP[j, :], RC_naive_AR, EL_AR_K_HP[j, :],
                                                              EL_naive_AR, RC_AR_K_CI_HP[j, :, :],
                                                              EL_AR_K_CI_HP[j, :, :])

    # Independent case
    # Montecarlo
    EL_vec_ind_SG_HP, EL_ind_SG_HP[j, :], RC_ind_SG_HP[j, :], RC_ind_SG_CI_HP[j, :, :], EL_SG_ind_CI_HP[j, :,:] = u.montecarlo_RC_HP(N1,
                                                                                      k_hat_SG,sigma_SG,LGD_hat,
                                                                                      sigma_LGD,alpha, g1,
                                                                                      M1,epsilon,3,N_obligors[j],corr_SG)
    EL_vec_ind_AR_HP, EL_ind_AR_HP[j, :], RC_ind_AR_HP[j, :], RC_ind_AR_CI_HP[j, :, :], EL_AR_ind_CI_HP[j, :,:] = u.montecarlo_RC_HP(N1,
                                                                                      k_hat_AR,sigma_AR,LGD_hat,
                                                                                      sigma_LGD,alpha, g1,
                                                                                      M1,epsilon,3,N_obligors[j],corr_AR)
    # Add-on
    Add_on_ind_SG_HP[j, :], Add_on_ind_SG_CI_HP[j, :, :] = u.Add_on(RC_ind_SG_HP[j, :], RC_naive_SG, EL_ind_SG_HP[j, :],
                                                                  EL_naive_SG, RC_ind_SG_CI_HP[j, :, :],
                                                                  EL_SG_ind_CI_HP[j, :, :])
    Add_on_ind_AR_HP[j, :], Add_on_ind_AR_CI_HP[j, :, :] = u.Add_on(RC_ind_AR_HP[j, :], RC_naive_AR, EL_ind_AR_HP[j, :],
                                                                  EL_naive_AR, RC_ind_AR_CI_HP[j, :, :],
                                                                  EL_AR_ind_CI_HP[j, :, :])

    # Dependent case
    # Montecarlo
    EL_vec_dep_SG_HP, EL_dep_SG_HP[j, :], RC_dep_SG_HP[j, :], RC_dep_SG_CI_HP[j, :, :], EL_SG_dep_CI_HP[j, :,:] = u.montecarlo_RC_HP(N1,
                                                                                       k_hat_SG,sigma_SG,LGD_hat,
                                                                                       sigma_LGD,alpha, g1,
                                                                                       M1,epsilon,4,N_obligors[j],corr_SG)
    EL_vec_dep_AR_HP, EL_dep_AR_HP[j, :], RC_dep_AR_HP[j, :], RC_dep_AR_CI_HP[j, :, :], EL_AR_dep_CI_HP[j, :,:] = u.montecarlo_RC_HP(N1,
                                                                                       k_hat_AR,sigma_AR,LGD_hat,
                                                                                       sigma_LGD,alpha, g1,
                                                                                       M1,epsilon,4,N_obligors[j],corr_AR)

    # Add-on
    Add_on_dep_SG_HP[j, :], Add_on_dep_SG_CI_HP[j, :, :] = u.Add_on(RC_dep_SG_HP[j, :], RC_naive_SG, EL_dep_SG_HP[j, :],
                                                                  EL_naive_SG, RC_dep_SG_CI_HP[j, :, :],
                                                                  EL_SG_dep_CI_HP[j, :, :])
    Add_on_dep_AR_HP[j, :], Add_on_dep_AR_CI_HP[j, :, :] = u.Add_on(RC_dep_AR_HP[j, :], RC_naive_AR, EL_dep_AR_HP[j, :],
                                                                  EL_naive_AR, RC_dep_AR_CI_HP[j, :, :],
                                                                  EL_AR_dep_CI_HP[j, :, :])
    for i in range(len(alpha)):
        print("\n")
        print(f"Add-on (alpha = {alpha[i]:.3f} - N obligors = {N_obligors[j]})\n\n")
        print(f"Only LGD simulation:                SG = {Add_on_SG_LGD_HP[j, i]:.4f} - CI = [{Add_on_SG_LGD_CI_HP[j, i, 0]:.4f}, {Add_on_SG_LGD_CI_HP[j, i, 1]:.4f}]   AR = {Add_on_AR_LGD_HP[j, i]:.4f} - CI = [{Add_on_AR_LGD_CI_HP[j, i, 0]:.4f}, {Add_on_AR_LGD_CI_HP[j, i, 1]:.4f}]\n")
        print(f"Only K simulation:                  SG = {Add_on_SG_K_HP[j, i]:.4f} - CI = [{Add_on_SG_K_CI_HP[j, i, 0]:.4f}, {Add_on_SG_K_CI_HP[j, i, 1]:.4f}]   AR = {Add_on_AR_K_HP[j, i]:.4f} - CI = [{Add_on_AR_K_CI_HP[j, i, 0]:.4f}, {Add_on_AR_K_CI_HP[j, i, 1]:.4f}]\n")
        print(f"LGD - K independent simulation:     SG = {Add_on_ind_SG_HP[j, i]:.4f} - CI = [{Add_on_ind_SG_CI_HP[j, i, 0]:.4f}, {Add_on_ind_SG_CI_HP[j, i, 1]:.4f}]   AR = {Add_on_ind_AR_HP[j, i]:.4f} - CI = [{Add_on_ind_AR_CI_HP[j, i, 0]:.4f}, {Add_on_ind_AR_CI_HP[j, i, 1]:.4f}]\n")
        print(f"LGD - K dependent simulation:       SG = {Add_on_dep_SG_HP[j, i]:.4f} - CI = [{Add_on_dep_SG_CI_HP[j, i, 0]:.4f}, {Add_on_dep_SG_CI_HP[j, i, 1]:.4f}]   AR = {Add_on_dep_AR_HP[j, i]:.4f} - CI = [{Add_on_dep_AR_CI_HP[j, i, 0]:.4f}, {Add_on_dep_AR_CI_HP[j, i, 1]:.4f}]\n\n")

    toc = time.time() - tic
    print(f"Montecarlo simulations computation time (N obligors = {N_obligors[j]}): {toc: .2f} seconds")

# Plotting
for i in range(len(alpha)):
    plt.figure()

    plt.subplot(4, 2, 1)
    plt.plot(N_obligors, RC_AR_LGD_HP[:, i], linewidth=2)
    plt.plot(N_obligors, RC_AR_LGD[i] * np.ones_like(N_obligors), linewidth=2)
    plt.title('Simulating LGD only - AR case')
    plt.xlabel('obligors')
    plt.ylabel('RC')

    plt.subplot(4, 2, 2)
    plt.plot(N_obligors, RC_SG_LGD_HP[:, i], linewidth=2)
    plt.plot(N_obligors, RC_SG_LGD[i] * np.ones_like(N_obligors), linewidth=2)
    plt.title('Simulating LGD only - SG case')
    plt.xlabel('obligors')
    plt.ylabel('RC')

    plt.subplot(4, 2, 3)
    plt.plot(N_obligors, RC_AR_K_HP[:, i], linewidth=2)
    plt.plot(N_obligors, RC_AR_K[i] * np.ones_like(N_obligors), linewidth=2)
    plt.title('Simulating K only - AR case')
    plt.xlabel('obligors')
    plt.ylabel('RC')

    plt.subplot(4, 2, 4)
    plt.plot(N_obligors, RC_SG_K_HP[:, i], linewidth=2)
    plt.plot(N_obligors, RC_SG_K[i] * np.ones_like(N_obligors), linewidth=2)
    plt.title('Simulating K only - SG case')
    plt.xlabel('obligors')
    plt.ylabel('RC')

    plt.subplot(4, 2, 5)
    plt.plot(N_obligors, RC_ind_AR_HP[:, i], linewidth=2)
    plt.plot(N_obligors, RC_ind_AR[i] * np.ones_like(N_obligors), linewidth=2)
    plt.title('Independent - AR case')
    plt.xlabel('obligors')
    plt.ylabel('RC')

    plt.subplot(4, 2, 6)
    plt.plot(N_obligors, RC_ind_SG_HP[:, i], linewidth=2)
    plt.plot(N_obligors, RC_ind_SG[i] * np.ones_like(N_obligors), linewidth=2)
    plt.title('Independent - SG case')
    plt.xlabel('obligors')
    plt.ylabel('RC')

    plt.subplot(4, 2, 7)
    plt.plot(N_obligors, RC_dep_AR_HP[:, i], linewidth=2)
    plt.plot(N_obligors, RC_dep_AR[i] * np.ones_like(N_obligors), linewidth=2)
    plt.title('Dependent - AR case')
    plt.xlabel('obligors')
    plt.ylabel('RC')

    plt.subplot(4, 2, 8)
    plt.plot(N_obligors, RC_dep_SG_HP[:, i], linewidth=2)
    plt.plot(N_obligors, RC_dep_SG[i] * np.ones_like(N_obligors), linewidth=2)
    plt.title('Dependent - SG case')
    plt.xlabel('obligors')
    plt.ylabel('RC')

plt.show()

print('\n--------------- DOUBLE t-STUDENT CASE ---------------\n')

# Market parameters simulation
np.random.seed(seed_m)
M = np.random.randn(N)

# Initialize arrays
EL_SG_LGD_T_double = np.zeros((len(alpha), 19))
RC_SG_LGD_T_double = np.zeros((len(alpha), 19))
RC_SG_LGD_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_AR_LGD_T_double = np.zeros((len(alpha), 19))
RC_AR_LGD_T_double = np.zeros((len(alpha), 19))
RC_AR_LGD_CI_T_double = np.zeros((len(alpha), 19, 2))
Add_on_SG_LGD_T_double = np.zeros((len(alpha), 19))
Add_on_SG_LGD_CI_T_double = np.zeros((len(alpha), 19, 2))
Add_on_AR_LGD_T_double = np.zeros((len(alpha), 19))
Add_on_AR_LGD_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_SG_K_T_double = np.zeros((len(alpha), 19))
RC_SG_K_T_double = np.zeros((len(alpha), 19))
RC_SG_K_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_AR_K_T_double = np.zeros((len(alpha), 19))
RC_AR_K_T_double = np.zeros((len(alpha), 19))
RC_AR_K_CI_T_double = np.zeros((len(alpha), 19, 2))
Add_on_SG_K_T_double = np.zeros((len(alpha), 19))
Add_on_SG_K_CI_T_double = np.zeros((len(alpha), 19, 2))
Add_on_AR_K_T_double = np.zeros((len(alpha), 19))
Add_on_AR_K_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_ind_SG_T_double = np.zeros((len(alpha), 19))
RC_ind_SG_T_double = np.zeros((len(alpha), 19))
RC_ind_SG_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_ind_AR_T_double = np.zeros((len(alpha), 19))
RC_ind_AR_T_double = np.zeros((len(alpha), 19))
RC_ind_AR_CI_T_double = np.zeros((len(alpha), 19, 2))
Add_on_ind_SG_T_double = np.zeros((len(alpha), 19))
Add_on_ind_SG_CI_T_double = np.zeros((len(alpha), 19, 2))
Add_on_ind_AR_T_double = np.zeros((len(alpha), 19))
Add_on_ind_AR_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_dep_SG_T_double = np.zeros((len(alpha), 19))
RC_dep_SG_T_double = np.zeros((len(alpha), 19))
RC_dep_SG_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_dep_AR_T_double = np.zeros((len(alpha), 19))
RC_dep_AR_T_double = np.zeros((len(alpha), 19))
RC_dep_AR_CI_T_double = np.zeros((len(alpha), 19, 2))
Add_on_dep_SG_T_double = np.zeros((len(alpha), 19))
Add_on_dep_SG_CI_T_double = np.zeros((len(alpha), 19, 2))
Add_on_dep_AR_T_double = np.zeros((len(alpha), 19))
Add_on_dep_AR_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_SG_LGD_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_SG_K_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_ind_SG_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_dep_SG_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_AR_LGD_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_AR_K_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_ind_AR_CI_T_double = np.zeros((len(alpha), 19, 2))
EL_dep_AR_CI_T_double = np.zeros((len(alpha), 19, 2))

for dof in range(2, 21):
    tic = time.time()
    np.random.seed(seed)

    # Double t-student parameters
    t1 = np.random.standard_t(dof, size=(2, N))
    t2 = np.random.standard_t(dof, size=(2, N))

    # LGD simulation, fixing K
    EL_vec_SG_LGD_T_double, EL_SG_LGD_T_double[:, dof - 2], RC_SG_LGD_T_double[:, dof - 2], \
    RC_SG_LGD_CI_T_double[:, dof - 2, :], EL_SG_LGD_CI_T_double[:, dof - 2, :] = \
        u.montecarlo_RC_T_double(N, norm.ppf(PD_SG_hat), sigma_SG, LGD_hat, sigma_LGD, alpha, t1, t2, M, 1,
                               corr_SG)
    EL_vec_AR_LGD_T_double, EL_AR_LGD_T_double[:, dof - 2], RC_AR_LGD_T_double[:, dof - 2], \
    RC_AR_LGD_CI_T_double[:, dof - 2, :], EL_AR_LGD_CI_T_double[:, dof - 2, :] = \
        u.montecarlo_RC_T_double(N, norm.ppf(PD_AR_hat), sigma_AR, LGD_hat, sigma_LGD, alpha, t1, t2, M, 1,
                               corr_AR)

    # Add-on computation
    Add_on_SG_LGD_T_double[:, dof - 2], Add_on_SG_LGD_CI_T_double[:, dof - 2, :] = u.Add_on(
        RC_SG_LGD_T_double[:, dof - 2], RC_naive_SG, EL_SG_LGD_T_double[:, dof - 2], EL_naive_SG,
        RC_SG_LGD_CI_T_double[:, dof - 2, :], EL_SG_LGD_CI_T_double[:, dof - 2, :])
    Add_on_AR_LGD_T_double[:, dof - 2], Add_on_AR_LGD_CI_T_double[:, dof - 2, :] = u.Add_on(
        RC_AR_LGD_T_double[:, dof - 2], RC_naive_AR, EL_AR_LGD_T_double[:, dof - 2], EL_naive_AR,
        RC_AR_LGD_CI_T_double[:, dof - 2, :], EL_AR_LGD_CI_T_double[:, dof - 2, :])

    # K simulation, fixing LGD
    EL_vec_SG_K_T_double, EL_SG_K_T_double[:, dof - 2], RC_SG_K_T_double[:, dof - 2], \
    RC_SG_K_CI_T_double[:, dof - 2, :], EL_SG_K_CI_T_double[:, dof - 2, :] = \
        u.montecarlo_RC_T_double(N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha, t1, t2, M, 2,
                               corr_SG)
    EL_vec_AR_K_T_double, EL_AR_K_T_double[:, dof - 2], RC_AR_K_T_double[:, dof - 2], \
    RC_AR_K_CI_T_double[:, dof - 2, :], EL_AR_K_CI_T_double[:, dof - 2, :] = \
        u.montecarlo_RC_T_double(N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha, t1, t2, M, 2,
                               corr_AR)
    # Add-on computation
    Add_on_SG_K_T_double[:, dof - 2], Add_on_SG_K_CI_T_double[:, dof - 2, :] = u.Add_on(
        RC_SG_K_T_double[:, dof - 2], RC_naive_SG, EL_SG_K_T_double[:, dof - 2], EL_naive_SG,
        RC_SG_K_CI_T_double[:, dof - 2, :], EL_SG_K_CI_T_double[:, dof - 2, :])
    Add_on_AR_K_T_double[:, dof - 2], Add_on_AR_K_CI_T_double[:, dof - 2, :] = u.Add_on(
        RC_AR_K_T_double[:, dof - 2], RC_naive_AR, EL_AR_K_T_double[:, dof - 2], EL_naive_AR,
        RC_AR_K_CI_T_double[:, dof - 2, :], EL_AR_K_CI_T_double[:, dof - 2, :])

    # K and LGD independent
    EL_vec_ind_SG_T_double, EL_ind_SG_T_double[:, dof - 2], RC_ind_SG_T_double[:, dof - 2], \
        RC_ind_SG_CI_T_double[:, dof - 2, :], EL_ind_SG_CI_T_double[:, dof - 2, :] = \
        u.montecarlo_RC_T_double(N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha, t1, t2, M, 3,
                                 corr_SG)
    EL_vec_ind_AR_T_double, EL_ind_AR_T_double[:, dof - 2], RC_ind_AR_T_double[:, dof - 2], \
        RC_ind_AR_CI_T_double[:, dof - 2, :], EL_ind_AR_CI_T_double[:, dof - 2, :] = \
        u.montecarlo_RC_T_double(N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha, t1, t2, M, 3,
                                 corr_AR)

    # Add-on computation
    Add_on_ind_SG_T_double[:, dof - 2], Add_on_ind_SG_CI_T_double[:, dof - 2, :] = u.Add_on(
        RC_ind_SG_T_double[:, dof - 2], RC_naive_SG, EL_ind_SG_T_double[:, dof - 2], EL_naive_SG,
        RC_ind_SG_CI_T_double[:, dof - 2, :], EL_ind_SG_CI_T_double[:, dof - 2, :])
    Add_on_ind_AR_T_double[:, dof - 2], Add_on_ind_AR_CI_T_double[:, dof - 2, :] = u.Add_on(
        RC_ind_AR_T_double[:, dof - 2], RC_naive_AR, EL_ind_AR_T_double[:, dof - 2], EL_naive_AR,
        RC_ind_AR_CI_T_double[:, dof - 2, :], EL_ind_AR_CI_T_double[:, dof - 2, :])

    # K and LGD dependent
    EL_vec_dep_SG_T_double, EL_dep_SG_T_double[:, dof - 2], RC_dep_SG_T_double[:, dof - 2], \
        RC_dep_SG_CI_T_double[:, dof - 2, :], EL_dep_SG_CI_T_double[:, dof - 2, :] = \
        u.montecarlo_RC_T_double(N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha, t1, t2, M, 4,
                                 corr_SG)

    EL_vec_dep_AR_T_double, EL_dep_AR_T_double[:, dof - 2], RC_dep_AR_T_double[:, dof - 2], \
        RC_dep_AR_CI_T_double[:, dof - 2, :], EL_dep_AR_CI_T_double[:, dof - 2, :] = \
        u.montecarlo_RC_T_double(N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha, t1, t2, M, 4,
                                 corr_AR)

    # Add-on computation
    Add_on_dep_SG_T_double[:, dof - 2], Add_on_dep_SG_CI_T_double[:, dof - 2, :] = u.Add_on(
        RC_dep_SG_T_double[:, dof - 2], RC_naive_SG, EL_dep_SG_T_double[:, dof - 2], EL_naive_SG,
        RC_dep_SG_CI_T_double[:, dof - 2, :], EL_dep_SG_CI_T_double[:, dof - 2, :])
    Add_on_dep_AR_T_double[:, dof - 2], Add_on_dep_AR_CI_T_double[:, dof - 2, :] = u.Add_on(
        RC_dep_AR_T_double[:, dof - 2], RC_naive_AR, EL_dep_AR_T_double[:, dof - 2], EL_naive_AR,
        RC_dep_AR_CI_T_double[:, dof - 2, :], EL_dep_AR_CI_T_double[:, dof - 2, :])

    for i in range(len(alpha)):
        print("\n")
        print(f"Add-on (alpha = {alpha[i]:.3f} - dof = {dof})\n")
        print(f"Only LGD simulation:              SG = {Add_on_SG_LGD_T_double[i, dof - 2]:.4f} - "
              f"CI = [{Add_on_SG_LGD_CI_T_double[i, dof - 2, 0]:.4f}, {Add_on_SG_LGD_CI_T_double[i, dof - 2, 1]:.4f}]   "
              f"AR = {Add_on_AR_LGD_T_double[i, dof - 2]:.4f} - "
              f"CI = [{Add_on_AR_LGD_CI_T_double[i, dof - 2, 0]:.4f}, {Add_on_AR_LGD_CI_T_double[i, dof - 2, 1]:.4f}]")
        print(f"Only K simulation:                SG = {Add_on_SG_K_T_double[i, dof - 2]:.4f} - "
              f"CI = [{Add_on_SG_K_CI_T_double[i, dof - 2, 0]:.4f}, {Add_on_SG_K_CI_T_double[i, dof - 2, 1]:.4f}]   "
              f"AR = {Add_on_AR_K_T_double[i, dof - 2]:.4f} - "
              f"CI = [{Add_on_AR_K_CI_T_double[i, dof - 2, 0]:.4f}, {Add_on_AR_K_CI_T_double[i, dof - 2, 1]:.4f}]")
        print(f"LGD - K independent simulation:   SG = {Add_on_ind_SG_T_double[i, dof - 2]:.4f} - "
              f"CI = [{Add_on_ind_SG_CI_T_double[i, dof - 2, 0]:.4f}, {Add_on_ind_SG_CI_T_double[i, dof - 2, 1]:.4f}]   "
              f"AR = {Add_on_ind_AR_T_double[i, dof - 2]:.4f} - "
              f"CI = [{Add_on_ind_AR_CI_T_double[i, dof - 2, 0]:.4f}, {Add_on_ind_AR_CI_T_double[i, dof - 2, 1]:.4f}]")
        print(f"LGD - K dependent simulation:     SG = {Add_on_dep_SG_T_double[i, dof - 2]:.4f} - "
              f"CI = [{Add_on_dep_SG_CI_T_double[i, dof - 2, 0]:.4f}, {Add_on_dep_SG_CI_T_double[i, dof - 2, 1]:.4f}]   "
              f"AR = {Add_on_dep_AR_T_double[i, dof - 2]:.4f} - "
              f"CI = [{Add_on_dep_AR_CI_T_double[i, dof - 2, 0]:.4f}, {Add_on_dep_AR_CI_T_double[i, dof - 2, 1]:.4f}]")

        toc = time.time() - tic
        print(f"Montecarlo simulations computation time (dof = {dof}): {toc:.2f} seconds")

x = np.arange(2, 21)

# Plotting
for i in range(len(alpha)):
    plt.figure()

    plt.subplot(4, 2, 1)
    plt.plot(x, RC_AR_LGD_T_double[i], linewidth=2)
    plt.plot(x, RC_AR_LGD[i] * np.ones_like(x), linewidth=2)
    plt.title('Simulating LGD only - AR case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 2)
    plt.plot(x, RC_SG_LGD_T_double[i], linewidth=2)
    plt.plot(x, RC_SG_LGD[i] * np.ones_like(x), linewidth=2)
    plt.title('Simulating LGD only - SG case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 3)
    plt.plot(x, RC_AR_K_T_double[i], linewidth=2)
    plt.plot(x, RC_AR_K[i] * np.ones_like(x), linewidth=2)
    plt.title('Simulating K only - AR case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 4)
    plt.plot(x, RC_SG_K_T_double[i], linewidth=2)
    plt.plot(x, RC_SG_K[i] * np.ones_like(x), linewidth=2)
    plt.title('Simulating K only - SG case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 5)
    plt.plot(x, RC_ind_AR_T_double[i], linewidth=2)
    plt.plot(x, RC_ind_AR[i] * np.ones_like(x), linewidth=2)
    plt.title('Independent - AR case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 6)
    plt.plot(x, RC_ind_SG_T_double[i], linewidth=2)
    plt.plot(x, RC_ind_SG[i] * np.ones_like(x), linewidth=2)
    plt.title('Independent - SG case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 7)
    plt.plot(x, RC_dep_AR_T_double[i], linewidth=2)
    plt.plot(x, RC_dep_AR[i] * np.ones_like(x), linewidth=2)
    plt.title('Dependent - AR case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 8)
    plt.plot(x, RC_dep_SG_T_double[i], linewidth=2)
    plt.plot(x, RC_dep_SG[i] * np.ones_like(x), linewidth=2)
    plt.title('Dependent - SG case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

plt.show()

# Montecarlo simulation N = 50 double t-student

# Initialize variables
EL_vec_SG_LGD_HP_T_double = np.zeros((19, len(alpha)))
EL_SG_LGD_HP_T_double = np.zeros((19, len(alpha)))
RC_SG_LGD_HP_T_double = np.zeros((19, len(alpha)))
RC_SG_LGD_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_SG_LGD_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_vec_AR_LGD_HP_T_double = np.zeros((19, len(alpha)))
EL_AR_LGD_HP_T_double = np.zeros((19, len(alpha)))
RC_AR_LGD_HP_T_double = np.zeros((19, len(alpha)))
RC_AR_LGD_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_AR_LGD_CI_HP_T_double = np.zeros((19, len(alpha), 2))

Add_on_SG_LGD_HP_T_double = np.zeros((19, len(alpha)))
Add_on_SG_LGD_CI_HP_T_double = np.zeros((19, len(alpha), 2))
Add_on_AR_LGD_HP_T_double = np.zeros((19, len(alpha)))
Add_on_AR_LGD_CI_HP_T_double = np.zeros((19, len(alpha), 2))

EL_vec_SG_K_HP_T_double = np.zeros((19, len(alpha)))
EL_SG_K_HP_T_double = np.zeros((19, len(alpha)))
RC_SG_K_HP_T_double = np.zeros((19, len(alpha)))
RC_SG_K_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_SG_K_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_vec_AR_K_HP_T_double = np.zeros((19, len(alpha)))
EL_AR_K_HP_T_double = np.zeros((19, len(alpha)))
RC_AR_K_HP_T_double = np.zeros((19, len(alpha)))
RC_AR_K_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_AR_K_CI_HP_T_double = np.zeros((19, len(alpha), 2))

Add_on_SG_K_HP_T_double = np.zeros((19, len(alpha)))
Add_on_SG_K_CI_HP_T_double = np.zeros((19, len(alpha), 2))
Add_on_AR_K_HP_T_double = np.zeros((19, len(alpha)))
Add_on_AR_K_CI_HP_T_double = np.zeros((19, len(alpha), 2))

EL_vec_ind_SG_HP_T_double = np.zeros((19, len(alpha)))
EL_ind_SG_HP_T_double = np.zeros((19, len(alpha)))
RC_ind_SG_HP_T_double = np.zeros((19, len(alpha)))
RC_ind_SG_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_SG_ind_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_vec_ind_AR_HP_T_double = np.zeros((19, len(alpha)))
EL_ind_AR_HP_T_double = np.zeros((19, len(alpha)))
RC_ind_AR_HP_T_double = np.zeros((19, len(alpha)))
RC_ind_AR_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_AR_ind_CI_HP_T_double = np.zeros((19, len(alpha), 2))

Add_on_ind_SG_HP_T_double = np.zeros((19, len(alpha)))
Add_on_ind_SG_CI_HP_T_double = np.zeros((19, len(alpha), 2))
Add_on_ind_AR_HP_T_double = np.zeros((19, len(alpha)))
Add_on_ind_AR_CI_HP_T_double = np.zeros((19, len(alpha), 2))

EL_vec_dep_SG_HP_T_double = np.zeros((19, len(alpha)))
EL_dep_SG_HP_T_double = np.zeros((19, len(alpha)))
RC_dep_SG_HP_T_double = np.zeros((19, len(alpha)))
RC_dep_SG_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_SG_dep_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_vec_dep_AR_HP_T_double = np.zeros((19, len(alpha)))
EL_dep_AR_HP_T_double = np.zeros((19, len(alpha)))
RC_dep_AR_HP_T_double = np.zeros((19, len(alpha)))
RC_dep_AR_CI_HP_T_double = np.zeros((19, len(alpha), 2))
EL_AR_dep_CI_HP_T_double = np.zeros((19, len(alpha), 2))

Add_on_dep_SG_HP_T_double = np.zeros((19, len(alpha)))
Add_on_dep_SG_CI_HP_T_double = np.zeros((19, len(alpha), 2))
Add_on_dep_AR_HP_T_double = np.zeros((19, len(alpha)))
Add_on_dep_AR_CI_HP_T_double = np.zeros((19, len(alpha), 2))

print("\n------------------ Montecarlo Regulatory capital HP Stress Double t-student - Add on computation -----------------\n\n")

for dof in range(2, 21):
    np.random.seed(seed)

    # Idiosinchratic terms simulation
    epsilon = np.random.randn(N_obligors[0], N1)  # idiosinchratic terms simulation
    t1_T = np.random.standard_t(dof, size=(2, N1))
    t2_T = np.random.standard_t(dof, size=(2, N1))

    tic = time.time()

    # Fix K, only LGD simulation
    # Montecarlo
    (EL_vec_SG_LGD_HP_T_double, EL_SG_LGD_HP_T_double[dof - 2, :], RC_SG_LGD_HP_T_double[dof - 2,:],
                                                                      RC_SG_LGD_CI_HP_T_double[dof - 2, :, :],
                                                                      EL_SG_LGD_CI_HP_T_double[dof - 2, :, :]) \
                                                                      = u.montecarlo_RC_HP_T_double(N1, norm.ppf(
                                                                      PD_SG_hat), sigma_SG, LGD_hat, sigma_LGD, alpha, t1_T, t2_T, M1, epsilon, 1, N_obligors[0], corr_SG)
    (EL_vec_AR_LGD_HP_T_double, EL_AR_LGD_HP_T_double[dof - 2, :], RC_AR_LGD_HP_T_double[dof - 2,:],
                                                                      RC_AR_LGD_CI_HP_T_double[dof - 2, :,:],
                                                                      EL_AR_LGD_CI_HP_T_double[dof - 2, :,:]) \
                                                                      = u.montecarlo_RC_HP_T_double(N1, norm.ppf(
                                                                      PD_AR_hat), sigma_AR, LGD_hat, sigma_LGD, alpha, t1_T, t2_T, M1, epsilon, 1, N_obligors[0], corr_AR)

    # Add-on
    Add_on_SG_LGD_HP_T_double[dof - 2, :], Add_on_SG_LGD_CI_HP_T_double[dof - 2, :, :] = u.Add_on(
        RC_SG_LGD_HP_T_double[dof - 2, :], RC_naive_SG, EL_SG_LGD_HP_T_double[dof - 2, :], EL_naive_SG,
        np.squeeze(RC_SG_LGD_CI_HP_T_double[dof - 2, :, :]), np.squeeze(EL_SG_LGD_CI_HP_T_double[dof - 2, :, :]))
    Add_on_AR_LGD_HP_T_double[dof - 2, :], Add_on_AR_LGD_CI_HP_T_double[dof - 2, :, :] = u.Add_on(
        RC_AR_LGD_HP_T_double[dof - 2, :], RC_naive_AR, EL_AR_LGD_HP_T_double[dof - 2, :], EL_naive_AR,
        np.squeeze(RC_AR_LGD_CI_HP_T_double[dof - 2, :, :]), np.squeeze(EL_AR_LGD_CI_HP_T_double[dof - 2, :, :]))

    # Fix LGD, only K simulation
    # Montecarlo
    (EL_vec_SG_K_HP_T_double, EL_SG_K_HP_T_double[dof - 2, :], RC_SG_K_HP_T_double[dof - 2, :], RC_SG_K_CI_HP_T_double[dof - 2, :,:],
                                                                        EL_SG_K_CI_HP_T_double[dof - 2, :,:]) = u.montecarlo_RC_HP_T_double(N1, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha, t1_T, t2_T, M1, epsilon, 2, N_obligors[0],corr_SG)
    (EL_vec_AR_K_HP_T_double, EL_AR_K_HP_T_double[dof - 2, :], RC_AR_K_HP_T_double[dof - 2, :], RC_AR_K_CI_HP_T_double[dof - 2, :,:],
                                                                        EL_AR_K_CI_HP_T_double[dof - 2, :,:]) = u.montecarlo_RC_HP_T_double(N1, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha, t1_T, t2_T, M1, epsilon, 2, N_obligors[0], corr_AR)

    # Add-on
    Add_on_SG_K_HP_T_double[dof - 2, :], Add_on_SG_K_CI_HP_T_double[dof - 2, :, :] = u.Add_on(
        RC_SG_K_HP_T_double[dof - 2, :], RC_naive_SG, EL_SG_K_HP_T_double[dof - 2, :], EL_naive_SG,
        np.squeeze(RC_SG_K_CI_HP_T_double[dof - 2, :, :]), np.squeeze(EL_SG_K_CI_HP_T_double[dof - 2, :, :]))
    Add_on_AR_K_HP_T_double[dof - 2, :], Add_on_AR_K_CI_HP_T_double[dof - 2, :, :] = u.Add_on(
        RC_AR_K_HP_T_double[dof - 2, :], RC_naive_AR, EL_AR_K_HP_T_double[dof - 2, :], EL_naive_AR,
        np.squeeze(RC_AR_K_CI_HP_T_double[dof - 2, :, :]), np.squeeze(EL_AR_K_CI_HP_T_double[dof - 2, :, :]))

    # Independent case
    # Montecarlo
    (EL_vec_ind_SG_HP_T_double, EL_ind_SG_HP_T_double[dof - 2, :], RC_ind_SG_HP_T_double[dof - 2,:],
                                                                        RC_ind_SG_CI_HP_T_double[dof - 2, :,:], EL_SG_ind_CI_HP_T_double[dof - 2, :,:]) = u.montecarlo_RC_HP_T_double(N1, k_hat_SG,sigma_SG,
                                                                                                    LGD_hat,sigma_LGD,
                                                                                                    alpha, t1_T,t2_T, M1,
                                                                                                    epsilon, 3, N_obligors[0], corr_SG)
    (EL_vec_ind_AR_HP_T_double, EL_ind_AR_HP_T_double[dof - 2, :], RC_ind_AR_HP_T_double[dof - 2, :],
                                                                        RC_ind_AR_CI_HP_T_double[dof - 2, :, :], EL_AR_ind_CI_HP_T_double[dof - 2, :, :]) = u.montecarlo_RC_HP_T_double(N1, k_hat_AR, sigma_AR,
                                                                                                    LGD_hat, sigma_LGD,
                                                                                                    alpha, t1_T, t2_T, M1,
                                                                                                    epsilon, 3, N_obligors[0], corr_AR)

    # Add-on
    Add_on_ind_SG_HP_T_double[dof - 2, :], Add_on_ind_SG_CI_HP_T_double[dof - 2, :, :] = u.Add_on(
        RC_ind_SG_HP_T_double[dof - 2, :], RC_naive_SG, EL_ind_SG_HP_T_double[dof - 2, :], EL_naive_SG,
        np.squeeze(RC_ind_SG_CI_HP_T_double[dof - 2, :, :]), np.squeeze(EL_SG_ind_CI_HP_T_double[dof - 2, :, :]))
    Add_on_ind_AR_HP_T_double[dof - 2, :], Add_on_ind_AR_CI_HP_T_double[dof - 2, :, :] = u.Add_on(
        RC_ind_AR_HP_T_double[dof - 2, :], RC_naive_AR, EL_ind_AR_HP_T_double[dof - 2, :], EL_naive_AR,
        np.squeeze(RC_ind_AR_CI_HP_T_double[dof - 2, :, :]), np.squeeze(EL_AR_ind_CI_HP_T_double[dof - 2, :, :]))

    # Dependent case
    # Montecarlo
    (EL_vec_dep_SG_HP_T_double, EL_dep_SG_HP_T_double[dof - 2, :], RC_dep_SG_HP_T_double[dof - 2,:],
                                                                RC_dep_SG_CI_HP_T_double[dof - 2, :,:],
                                                                EL_SG_dep_CI_HP_T_double[dof - 2, :,:]) = (
                                                                u.montecarlo_RC_HP_T_double(N1, k_hat_SG,
                                                                sigma_SG,LGD_hat,sigma_LGD,alpha, t1_T,t2_T, M1,
                                                                epsilon, 4,N_obligors[0],corr_SG))
    (EL_vec_dep_AR_HP_T_double, EL_dep_AR_HP_T_double[dof - 2, :], RC_dep_AR_HP_T_double[dof - 2,:],
                                                                RC_dep_AR_CI_HP_T_double[dof - 2, :, :],
                                                                EL_AR_dep_CI_HP_T_double[dof - 2, :,:]) = (
                                                                u.montecarlo_RC_HP_T_double(N1, k_hat_AR,
                                                                sigma_AR,LGD_hat,sigma_LGD,alpha, t1_T,
                                                                t2_T, M1,epsilon, 4,N_obligors[0],corr_AR))

    # Add-on
    Add_on_dep_SG_HP_T_double[dof - 2, :], Add_on_dep_SG_CI_HP_T_double[dof - 2, :, :] = u.Add_on(
        RC_dep_SG_HP_T_double[dof - 2, :], RC_naive_SG, EL_dep_SG_HP_T_double[dof - 2, :], EL_naive_SG,
        np.squeeze(RC_dep_SG_CI_HP_T_double[dof - 2, :, :]), np.squeeze(EL_SG_dep_CI_HP_T_double[dof - 2, :, :]))
    Add_on_dep_AR_HP_T_double[dof - 2, :], Add_on_dep_AR_CI_HP_T_double[dof - 2, :, :] = u.Add_on(
        RC_dep_AR_HP_T_double[dof - 2, :], RC_naive_AR, EL_dep_AR_HP_T_double[dof - 2, :], EL_naive_AR,
        np.squeeze(RC_dep_AR_CI_HP_T_double[dof - 2, :, :]), np.squeeze(EL_AR_dep_CI_HP_T_double[dof - 2, :, :]))

    for i in range(len(alpha)):
        print("\n")
        print("Add-on (alpha = %.3f - dof = %d)\n" % (alpha[i], dof))
        print("Only LGD simulation:              SG = %.4f - CI = [%.4f, %.4f]   AR = %.4f - CI = [%.4f, %.4f]" %
              (Add_on_SG_LGD_T_double[i, dof - 2], Add_on_SG_LGD_CI_T_double[i, dof - 2, 0],
               Add_on_SG_LGD_CI_T_double[i, dof - 2, 1],
               Add_on_AR_LGD_T_double[i, dof - 2], Add_on_AR_LGD_CI_T_double[i, dof - 2, 0],
               Add_on_AR_LGD_CI_T_double[i, dof - 2, 1]))
        print("Only K simulation:                SG = %.4f - CI = [%.4f, %.4f]   AR = %.4f - CI = [%.4f, %.4f]" %
              (Add_on_SG_K_T_double[i, dof - 2], Add_on_SG_K_CI_T_double[i, dof - 2, 0],
               Add_on_SG_K_CI_T_double[i, dof - 2, 1],
               Add_on_AR_K_T_double[i, dof - 2], Add_on_AR_K_CI_T_double[i, dof - 2, 0],
               Add_on_AR_K_CI_T_double[i, dof - 2, 1]))
        print("LGD - K independent simulation:   SG = %.4f - CI = [%.4f, %.4f]   AR = %.4f - CI = [%.4f, %.4f]" %
              (Add_on_ind_SG_T_double[i, dof - 2], Add_on_ind_SG_CI_T_double[i, dof - 2, 0],
               Add_on_ind_SG_CI_T_double[i, dof - 2, 1],
               Add_on_ind_AR_T_double[i, dof - 2], Add_on_ind_AR_CI_T_double[i, dof - 2, 0],
               Add_on_ind_AR_CI_T_double[i, dof - 2, 1]))
        print("LGD - K dependent simulation:     SG = %.4f - CI = [%.4f, %.4f]   AR = %.4f - CI = [%.4f, %.4f]" %
              (Add_on_dep_SG_T_double[i, dof - 2], Add_on_dep_SG_CI_T_double[i, dof - 2, 0],
               Add_on_dep_SG_CI_T_double[i, dof - 2, 1],
               Add_on_dep_AR_T_double[i, dof - 2], Add_on_dep_AR_CI_T_double[i, dof - 2, 0],
               Add_on_dep_AR_CI_T_double[i, dof - 2, 1]))
    toc = time.time() - tic
    print(f"Montecarlo simulations computation time (N obligors = {N_obligors[0]} - dof = {dof}): {toc: .2f} seconds")


x = np.arange(2, 21)

for i in range(len(alpha)):
    plt.figure()

    plt.subplot(4, 2, 1)
    plt.plot(x, RC_AR_LGD_HP_T_double[:, i], linewidth=2)
    plt.plot(x, RC_AR_LGD_HP[0, i] * np.ones(len(x)), linewidth=2)
    plt.title('Simulating LGD only - AR case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 2)
    plt.plot(x, RC_SG_LGD_HP_T_double[:, i], linewidth=2)
    plt.plot(x, RC_SG_LGD_HP[0, i] * np.ones(len(x)), linewidth=2)
    plt.title('Simulating LGD only - SG case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 3)
    plt.plot(x, RC_AR_K_HP_T_double[:, i], linewidth=2)
    plt.plot(x, RC_AR_K_HP[0, i] * np.ones(len(x)), linewidth=2)
    plt.title('Simulating K only - AR case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 4)
    plt.plot(x, RC_SG_K_HP_T_double[:, i], linewidth=2)
    plt.plot(x, RC_SG_K_HP[0, i] * np.ones(len(x)), linewidth=2)
    plt.title('Simulating K only - SG case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 5)
    plt.plot(x, RC_ind_AR_HP_T_double[:, i], linewidth=2)
    plt.plot(x, RC_ind_AR_HP[0, i] * np.ones(len(x)), linewidth=2)
    plt.title('Simulating LGD and K independent - AR case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 6)
    plt.plot(x, RC_ind_SG_HP_T_double[:, i], linewidth=2)
    plt.plot(x, RC_ind_SG_HP[0, i] * np.ones(len(x)), linewidth=2)
    plt.title('Simulating LGD and K independent - SG case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 7)
    plt.plot(x, RC_dep_SG_HP_T_double[:, i], linewidth=2)
    plt.plot(x, RC_dep_SG_HP[0, i] * np.ones(len(x)), linewidth=2)
    plt.title('Simulating LGD and K dependent - AR case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

    plt.subplot(4, 2, 8)
    plt.plot(x, RC_dep_SG_HP_T_double[:, i], linewidth=2)
    plt.plot(x, RC_dep_SG_HP[0, i] * np.ones(len(x)), linewidth=2)
    plt.title('Simulating LGD and K dependent - SG case')
    plt.xlabel('degree of freedom')
    plt.ylabel('RC')

plt.show()

# Double t-student
print("\n --------------- t-STUDENT CASE ---------------\n")

# Initialize lists to store results
RC_SG_LGD_T = np.zeros(19)
RC_SG_LGD_CI_T = np.zeros((19, 2))
RC_AR_LGD_T = np.zeros(19)
RC_AR_LGD_CI_T = np.zeros((19, 2))
Add_on_SG_LGD_T = np.zeros(19)
Add_on_SG_LGD_CI_T = np.zeros((19, 2))
Add_on_AR_LGD_T = np.zeros(19)
Add_on_AR_LGD_CI_T = np.zeros((19, 2))

RC_SG_K_T = np.zeros(19)
RC_SG_K_CI_T = np.zeros((19, 2))
RC_AR_K_T = np.zeros(19)
RC_AR_K_CI_T = np.zeros((19, 2))
Add_on_SG_K_T = np.zeros(19)
Add_on_SG_K_CI_T = np.zeros((19, 2))
Add_on_AR_K_T = np.zeros(19)
Add_on_AR_K_CI_T = np.zeros((19, 2))

RC_ind_SG_T = np.zeros(19)
RC_ind_SG_CI_T = np.zeros((19, 2))
RC_ind_AR_T = np.zeros(19)
RC_ind_AR_CI_T = np.zeros((19, 2))
Add_on_ind_SG_T = np.zeros(19)
Add_on_ind_SG_CI_T = np.zeros((19, 2))
Add_on_ind_AR_T = np.zeros(19)
Add_on_ind_AR_CI_T = np.zeros((19, 2))

RC_dep_SG_T = np.zeros(19)
RC_dep_SG_CI_T = np.zeros((19, 2))
RC_dep_AR_T = np.zeros(19)
RC_dep_AR_CI_T = np.zeros((19, 2))
Add_on_dep_SG_T = np.zeros(19)
Add_on_dep_SG_CI_T = np.zeros((19, 2))
Add_on_dep_AR_T = np.zeros(19)
Add_on_dep_AR_CI_T = np.zeros((19, 2))

for i in range(19):
    start_time = time.time()
    dof = i + 2  # degrees of freedom

    # LGD and K t-Student simulation
    np.random.seed(seed)
    t = np.array([t_dist.rvs(dof, size=N), t_dist.rvs(dof, size=N)])

    # LGD simulation, fixing K
    EL_vec_SG_LGD_T, EL_SG_LGD_T, RC_SG_LGD_T[i], RC_SG_LGD_CI_T[i], EL_SG_LGD_CI_T = u.montecarlo_RC_T(
        N, norm.ppf(PD_SG_hat), sigma_SG, LGD_hat, sigma_LGD, alpha[0], t, M, 1, corr_SG
    )
    EL_vec_AR_LGD_T, EL_AR_LGD_T, RC_AR_LGD_T[i], RC_AR_LGD_CI_T[i], EL_AR_LGD_CI_T = u.montecarlo_RC_T(
        N, norm.ppf(PD_AR_hat), sigma_AR, LGD_hat, sigma_LGD, alpha[0], t, M, 1, corr_AR
    )

    # Add-on computation
    Add_on_SG_LGD_T[i], Add_on_SG_LGD_CI_T[i] = u.Add_on(
        RC_SG_LGD_T[i], RC_naive_SG[0], EL_SG_LGD_T, EL_naive_SG, RC_SG_LGD_CI_T[i], EL_SG_LGD_CI_T
    )
    Add_on_AR_LGD_T[i], Add_on_AR_LGD_CI_T[i] = u.Add_on(
        RC_AR_LGD_T[i], RC_naive_AR[0], EL_AR_LGD_T, EL_naive_AR, RC_AR_LGD_CI_T[i], EL_AR_LGD_CI_T
    )

    # K simulation, fixing LGD
    EL_vec_SG_K_T, EL_SG_K_T, RC_SG_K_T[i], RC_SG_K_CI_T[i], EL_SG_K_CI_T = u.montecarlo_RC_T(
        N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha[0], t, M, 2, corr_SG
    )
    EL_vec_AR_K_T, EL_AR_K_T, RC_AR_K_T[i], RC_AR_K_CI_T[i], EL_AR_K_CI_T = u.montecarlo_RC_T(
        N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha[0], t, M, 2, corr_AR
    )

    # Add-on computation
    Add_on_SG_K_T[i], Add_on_SG_K_CI_T[i] = u.Add_on(
        RC_SG_K_T[i], RC_naive_SG[0], EL_SG_K_T, EL_naive_SG, RC_SG_K_CI_T[i], EL_SG_K_CI_T
    )
    Add_on_AR_K_T[i], Add_on_AR_K_CI_T[i] = u.Add_on(
        RC_AR_K_T[i], RC_naive_AR[0], EL_AR_K_T, EL_naive_AR, RC_AR_K_CI_T[i], EL_AR_K_CI_T
    )

    # K and LGD independent
    EL_vec_ind_SG_T, EL_ind_SG_T, RC_ind_SG_T[i], RC_ind_SG_CI_T[i], EL_SG_ind_CI_T = u.montecarlo_RC_T(
        N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha[0], t, M, 3, corr_SG
    )
    EL_vec_ind_AR_T, EL_ind_AR_T, RC_ind_AR_T[i], RC_ind_AR_CI_T[i], EL_AR_ind_CI_T = u.montecarlo_RC_T(
        N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha[0], t, M, 3, corr_AR
    )

    # Add-on computation
    Add_on_ind_SG_T[i], Add_on_ind_SG_CI_T[i] = u.Add_on(
        RC_ind_SG_T[i], RC_naive_SG[0], EL_ind_SG_T, EL_naive_SG, RC_ind_SG_CI_T[i], EL_SG_ind_CI_T)
    Add_on_ind_AR_T[i], Add_on_ind_AR_CI_T[i] = u.Add_on(
        RC_ind_AR_T[i], RC_naive_AR[0], EL_ind_AR_T, EL_naive_AR, RC_ind_AR_CI_T[i], EL_AR_ind_CI_T)

    # K and LGD dependent
    EL_vec_dep_SG_T, EL_dep_SG_T, RC_dep_SG_T[i], RC_dep_SG_CI_T[i], EL_SG_dep_CI_T = u.montecarlo_RC_T(
        N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha[0], t, M, 4, corr_SG)
    EL_vec_dep_AR_T, EL_dep_AR_T, RC_dep_AR_T[i], RC_dep_AR_CI_T[i], EL_AR_dep_CI_T = u.montecarlo_RC_T(
        N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha[0], t, M, 4, corr_AR)

    # Add-on computation
    Add_on_dep_SG_T[i], Add_on_dep_SG_CI_T[i] = u.Add_on(
        RC_dep_SG_T[i], RC_naive_SG[0], EL_dep_SG_T, EL_naive_SG, RC_dep_SG_CI_T[i], EL_SG_dep_CI_T)
    Add_on_dep_AR_T[i], Add_on_dep_AR_CI_T[i] = u.Add_on(
        RC_dep_AR_T[i], RC_naive_AR[0], EL_dep_AR_T, EL_naive_AR, RC_dep_AR_CI_T[i], EL_AR_dep_CI_T)

    print(f"\nAdd-on (alpha = {alpha[0]:.3f} - dof = {dof})\n")
    print(
        f"Only LGD simulation:                SG = {Add_on_SG_LGD_T[i]:.4f} - CI = [{Add_on_SG_LGD_CI_T[i, 0]:.4f}, {Add_on_SG_LGD_CI_T[i, 1]:.4f}]   AR = {Add_on_AR_LGD_T[i]:.4f} - CI = [{Add_on_AR_LGD_CI_T[i, 0]:.4f}, {Add_on_AR_LGD_CI_T[i, 1]:.4f}]")
    print(
        f"Only K simulation:                  SG = {Add_on_SG_K_T[i]:.4f} - CI = [{Add_on_SG_K_CI_T[i, 0]:.4f}, {Add_on_SG_K_CI_T[i, 1]:.4f}]   AR = {Add_on_AR_K_T[i]:.4f} - CI = [{Add_on_AR_K_CI_T[i, 0]:.4f}, {Add_on_AR_K_CI_T[i, 1]:.4f}]")
    print(
        f"LGD - K independent simulation:     SG = {Add_on_ind_SG_T[i]:.4f} - CI = [{Add_on_ind_SG_CI_T[i, 0]:.4f}, {Add_on_ind_SG_CI_T[i, 1]:.4f}]   AR = {Add_on_ind_AR_T[i]:.4f} - CI = [{Add_on_ind_AR_CI_T[i, 0]:.4f}, {Add_on_ind_AR_CI_T[i, 1]:.4f}]")
    print(
        f"LGD - K dependent simulation:       SG = {Add_on_dep_SG_T[i]:.4f} - CI = [{Add_on_dep_SG_CI_T[i, 0]:.4f}, {Add_on_dep_SG_CI_T[i, 1]:.4f}]   AR = {Add_on_dep_AR_T[i]:.4f} - CI = [{Add_on_dep_AR_CI_T[i, 0]:.4f}, {Add_on_dep_AR_CI_T[i, 1]:.4f}]\n")

    toc =  time.time() - start_time
    print(f"Montecarlo simulations computation time (dof = {dof}): {toc: .2f} seconds")

print('\n --------------- t-STUDENT CASE ---------------\n')
EL_vec_SG_LGD_T = np.zeros((N, 19))
EL_SG_LGD_T = np.zeros(19)
RC_SG_LGD_T = np.zeros(19)
RC_SG_LGD_CI_T = np.zeros((19, 2))
EL_SG_LGD_CI_T = np.zeros(19)

EL_vec_AR_LGD_T = np.zeros((N, 19))
EL_AR_LGD_T = np.zeros(19)
RC_AR_LGD_T = np.zeros(19)
RC_AR_LGD_CI_T = np.zeros((19, 2))
EL_AR_LGD_CI_T = np.zeros(19)

Add_on_SG_LGD_T = np.zeros(19)
Add_on_SG_LGD_CI_T = np.zeros((19, 2))
Add_on_AR_LGD_T = np.zeros(19)
Add_on_AR_LGD_CI_T = np.zeros((19, 2))

EL_vec_SG_K_T = np.zeros((N, 19))
EL_SG_K_T = np.zeros(19)
RC_SG_K_T = np.zeros(19)
RC_SG_K_CI_T = np.zeros((19, 2))
EL_SG_K_CI_T = np.zeros(19)

EL_vec_AR_K_T = np.zeros((N, 19))
EL_AR_K_T = np.zeros(19)
RC_AR_K_T = np.zeros(19)
RC_AR_K_CI_T = np.zeros((19, 2))
EL_AR_K_CI_T = np.zeros(19)

Add_on_SG_K_T = np.zeros(19)
Add_on_SG_K_CI_T = np.zeros((19, 2))
Add_on_AR_K_T = np.zeros(19)
Add_on_AR_K_CI_T = np.zeros((19, 2))

EL_vec_ind_SG_T = np.zeros((N, 19))
EL_ind_SG_T = np.zeros(19)
RC_ind_SG_T = np.zeros(19)
RC_ind_SG_CI_T = np.zeros((19, 2))
EL_SG_ind_CI_T = np.zeros(19)

EL_vec_ind_AR_T = np.zeros((N, 19))
EL_ind_AR_T = np.zeros(19)
RC_ind_AR_T = np.zeros(19)
RC_ind_AR_CI_T = np.zeros((19, 2))
EL_AR_ind_CI_T = np.zeros(19)

Add_on_ind_SG_T = np.zeros(19)
Add_on_ind_SG_CI_T = np.zeros((19, 2))
Add_on_ind_AR_T = np.zeros(19)
Add_on_ind_AR_CI_T = np.zeros((19, 2))

EL_vec_dep_SG_T = np.zeros((N, 19))
EL_dep_SG_T = np.zeros(19)
RC_dep_SG_T = np.zeros(19)
RC_dep_SG_CI_T = np.zeros((19, 2))
EL_SG_dep_CI_T = np.zeros(19)

EL_vec_dep_AR_T = np.zeros((N, 19))
EL_dep_AR_T = np.zeros(19)
RC_dep_AR_T = np.zeros(19)
RC_dep_AR_CI_T = np.zeros((19, 2))
EL_AR_dep_CI_T = np.zeros(19)

Add_on_dep_SG_T = np.zeros(19)
Add_on_dep_SG_CI_T = np.zeros((19, 2))
Add_on_dep_AR_T = np.zeros(19)
Add_on_dep_AR_CI_T = np.zeros((19, 2))

# Montecarlo
for i in range(0, 19):  # from dof = 2 to dof = 20
    start_time = time.time()

    dof = i + 2  # degrees of freedom

    # LGD and K t-Student simulation
    np.random.seed(seed)
    t = t_dist.rvs(dof, size=(2, N))

    # LGD simulation, fixing K
    EL_vec_SG_LGD_T, EL_SG_LGD_T, RC_SG_LGD_T[i], RC_SG_LGD_CI_T[i, :], EL_SG_LGD_CI_T = u.montecarlo_RC_T(
        N, norm.ppf(PD_SG_hat), sigma_SG, LGD_hat, sigma_LGD, alpha[0], t, M, 1, corr_SG)
    EL_vec_AR_LGD_T, EL_AR_LGD_T, RC_AR_LGD_T[i], RC_AR_LGD_CI_T[i, :], EL_AR_LGD_CI_T = u.montecarlo_RC_T(
        N, norm.ppf(PD_AR_hat), sigma_AR, LGD_hat, sigma_LGD, alpha[0], t, M, 1, corr_AR)

    # Add-on computation
    Add_on_SG_LGD_T[i], Add_on_SG_LGD_CI_T[i, :] = u.Add_on(
        RC_SG_LGD_T[i], RC_naive_SG[0], EL_SG_LGD_T, EL_naive_SG, RC_SG_LGD_CI_T[i, :], EL_SG_LGD_CI_T)
    Add_on_AR_LGD_T[i], Add_on_AR_LGD_CI_T[i, :] = u.Add_on(
        RC_AR_LGD_T[i], RC_naive_AR[0], EL_AR_LGD_T, EL_naive_AR, RC_AR_LGD_CI_T[i, :], EL_AR_LGD_CI_T)

    # K simulation, fixing LGD
    EL_vec_SG_K_T, EL_SG_K_T, RC_SG_K_T[i], RC_SG_K_CI_T[i, :], EL_SG_K_CI_T = u.montecarlo_RC_T(
        N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha[0], t, M, 2, corr_SG)
    EL_vec_AR_K_T, EL_AR_K_T, RC_AR_K_T[i], RC_AR_K_CI_T[i, :], EL_AR_K_CI_T = u.montecarlo_RC_T(
        N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha[0], t, M, 2, corr_AR)

    # Add-on computation
    Add_on_SG_K_T[i], Add_on_SG_K_CI_T[i, :] = u.Add_on(
        RC_SG_K_T[i], RC_naive_SG[0], EL_SG_K_T, EL_naive_SG, RC_SG_K_CI_T[i, :], EL_SG_K_CI_T)
    Add_on_AR_K_T[i], Add_on_AR_K_CI_T[i, :] = u.Add_on(
        RC_AR_K_T[i], RC_naive_AR[0], EL_AR_K_T, EL_naive_AR, RC_AR_K_CI_T[i, :], EL_AR_K_CI_T)

    # K and LGD independent
    EL_vec_ind_SG_T, EL_ind_SG_T, RC_ind_SG_T[i], RC_ind_SG_CI_T[i, :], EL_SG_ind_CI_T = u.montecarlo_RC_T(
        N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha[0], t, M, 3, corr_SG)
    EL_vec_ind_AR_T, EL_ind_AR_T, RC_ind_AR_T[i], RC_ind_AR_CI_T[i, :], EL_AR_ind_CI_T = u.montecarlo_RC_T(
        N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha[0], t, M, 3, corr_AR)

    # Add-on computation
    Add_on_ind_SG_T[i], Add_on_ind_SG_CI_T[i, :] = u.Add_on(
        RC_ind_SG_T[i], RC_naive_SG[0], EL_ind_SG_T, EL_naive_SG, RC_ind_SG_CI_T[i, :], EL_SG_ind_CI_T)
    Add_on_ind_AR_T[i], Add_on_ind_AR_CI_T[i, :] = u.Add_on(
        RC_ind_AR_T[i], RC_naive_AR[0], EL_ind_AR_T, EL_naive_AR, RC_ind_AR_CI_T[i, :], EL_AR_ind_CI_T)

    # K and LGD dependent
    EL_vec_dep_SG_T, EL_dep_SG_T, RC_dep_SG_T[i], RC_dep_SG_CI_T[i, :], EL_SG_dep_CI_T = u.montecarlo_RC_T(
        N, k_hat_SG, sigma_SG, LGD_hat, sigma_LGD, alpha[0], t, M, 4, corr_SG)
    EL_vec_dep_AR_T, EL_dep_AR_T, RC_dep_AR_T[i], RC_dep_AR_CI_T[i, :], EL_AR_dep_CI_T = u.montecarlo_RC_T(
        N, k_hat_AR, sigma_AR, LGD_hat, sigma_LGD, alpha[0], t, M, 4, corr_AR)

    # Add-on computation
    Add_on_dep_SG_T[i], Add_on_dep_SG_CI_T[i, :] = u.Add_on(
        RC_dep_SG_T[i], RC_naive_SG[0], EL_dep_SG_T, EL_naive_SG, RC_dep_SG_CI_T[i, :], EL_SG_dep_CI_T)
    Add_on_dep_AR_T[i], Add_on_dep_AR_CI_T[i, :] = u.Add_on(
        RC_dep_AR_T[i], RC_naive_AR[0], EL_dep_AR_T, EL_naive_AR, RC_dep_AR_CI_T[i, :], EL_AR_dep_CI_T)

    print("\nAdd-on (alpha = %.3f - dof = %d)\n" % (alpha[0], dof))
    print("Only LGD simulation:                SG = %.4f - CI = [%.4f, %.4f]   AR = %.4f - CI = [%.4f, %.4f]" %
          (Add_on_SG_LGD_T[i], Add_on_SG_LGD_CI_T[i, 0], Add_on_SG_LGD_CI_T[i, 1],
           Add_on_AR_LGD_T[i], Add_on_AR_LGD_CI_T[i, 0], Add_on_AR_LGD_CI_T[i, 1]))
    print("Only K simulation:                  SG = %.4f - CI = [%.4f, %.4f]   AR = %.4f - CI = [%.4f, %.4f]" %
          (Add_on_SG_K_T[i], Add_on_SG_K_CI_T[i, 0], Add_on_SG_K_CI_T[i, 1],
           Add_on_AR_K_T[i], Add_on_AR_K_CI_T[i, 0], Add_on_AR_K_CI_T[i, 1]))
    print("LGD - K independent simulation:     SG = %.4f - CI = [%.4f, %.4f]   AR = %.4f - CI = [%.4f, %.4f]" %
          (Add_on_ind_SG_T[i], Add_on_ind_SG_CI_T[i, 0], Add_on_ind_SG_CI_T[i, 1],
           Add_on_ind_AR_T[i], Add_on_ind_AR_CI_T[i, 0], Add_on_ind_AR_CI_T[i, 1]))
    print("LGD - K dependent simulation:       SG = %.4f - CI = [%.4f, %.4f]   AR = %.4f - CI = [%.4f, %.4f]\n" %
          (Add_on_dep_SG_T[i], Add_on_dep_SG_CI_T[i, 0], Add_on_dep_SG_CI_T[i, 1],
           Add_on_dep_AR_T[i], Add_on_dep_AR_CI_T[i, 0], Add_on_dep_AR_CI_T[i, 1]))

    toc = time.time() - start_time
    print(f"Montecarlo simulations computation time (dof = {dof}): {toc: .2f} seconds")


plt.figure()

# Subplot 1
plt.subplot(4, 2, 1)
plt.plot(x, RC_AR_LGD_T, linewidth=2)
plt.plot(x, RC_AR_LGD[0] * np.ones(19), linewidth=2)
plt.title('Simulating LGD only - AR case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')

# Subplot 2
plt.subplot(4, 2, 2)
plt.plot(x, RC_SG_LGD_T, linewidth=2)
plt.plot(x, RC_SG_LGD[0] * np.ones(19), linewidth=2)
plt.title('Simulating LGD only - SG case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')

# Subplot 3
plt.subplot(4, 2, 3)
plt.plot(x, RC_AR_K_T, linewidth=2)
plt.plot(x, RC_AR_K[0] * np.ones(19), linewidth=2)
plt.title('Simulating K only - AR case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')

# Subplot 4
plt.subplot(4, 2, 4)
plt.plot(x, RC_SG_K_T, linewidth=2)
plt.plot(x, RC_SG_K[0] * np.ones(19), linewidth=2)
plt.title('Simulating K only - SG case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')

# Subplot 5
plt.subplot(4, 2, 5)
plt.plot(x, RC_ind_AR_T, linewidth=2)
plt.plot(x, RC_ind_AR[0] * np.ones(19), linewidth=2)
plt.title('Simulating LGD and K as independent - AR case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')

# Subplot 6
plt.subplot(4, 2, 6)
plt.plot(x, RC_ind_SG_T, linewidth=2)
plt.plot(x, RC_ind_SG[0] * np.ones(19), linewidth=2)
plt.title('Simulating LGD and K as independent - SG case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')

# Subplot 7
plt.subplot(4, 2, 7)
plt.plot(x, RC_dep_AR_T, linewidth=2)
plt.plot(x, RC_dep_AR[0] * np.ones(19), linewidth=2)
plt.title('Simulating LGD and K as dependent - AR case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')

# Subplot 8
plt.subplot(4, 2, 8)
plt.plot(x, RC_dep_SG_T, linewidth=2)
plt.plot(x, RC_dep_SG[0] * np.ones(19), linewidth=2)
plt.title('Simulating LGD and K as dependent - SG case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')

plt.show()

x = np.arange(2, 21)

# Comparison between t and double-t distributions

plt.figure()

# Subplot 1
plt.subplot(4, 2, 1)
plt.plot(x, RC_AR_LGD_T_double[0], linewidth=2)
plt.plot(x, RC_AR_LGD_T, linewidth=2)
plt.title('Simulating LGD only - AR case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')
plt.legend(['double t', 't'])

# Subplot 2
plt.subplot(4, 2, 2)
plt.plot(x, RC_SG_LGD_T_double[0], linewidth=2)
plt.plot(x, RC_SG_LGD_T, linewidth=2)
plt.title('Simulating LGD only - SG case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')
plt.legend(['double t', 't'])

# Subplot 3
plt.subplot(4, 2, 3)
plt.plot(x, RC_AR_K_T_double[0], linewidth=2)
plt.plot(x, RC_AR_K_T, linewidth=2)
plt.title('Simulating K only - AR case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')
plt.legend(['double t', 't'])

# Subplot 4
plt.subplot(4, 2, 4)
plt.plot(x, RC_SG_K_T_double[0], linewidth=2)
plt.plot(x, RC_SG_K_T, linewidth=2)
plt.title('Simulating K only - SG case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')
plt.legend(['double t', 't'])

# Subplot 5
plt.subplot(4, 2, 5)
plt.plot(x, RC_ind_AR_T_double[0], linewidth=2)
plt.plot(x, RC_ind_AR_T, linewidth=2)
plt.title('Simulating LGD and K independent - AR case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')
plt.legend(['double t', 't'])

# Subplot 6
plt.subplot(4, 2, 6)
plt.plot(x, RC_ind_SG_T_double[0], linewidth=2)
plt.plot(x, RC_ind_SG_T, linewidth=2)
plt.title('Simulating LGD and K independent - SG case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')
plt.legend(['double t', 't'])

# Subplot 7
plt.subplot(4, 2, 7)
plt.plot(x, RC_dep_SG_T_double[0], linewidth=2)
plt.plot(x, RC_dep_SG_T, linewidth=2)
plt.title('Simulating LGD and K dependent - AR case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')
plt.legend(['double t', 't'])

# Subplot 8
plt.subplot(4, 2, 8)
plt.plot(x, RC_dep_SG_T_double[0], linewidth=2)
plt.plot(x, RC_dep_SG_T, linewidth=2)
plt.title('Simulating LGD and K dependent - SG case')
plt.xlabel('degree of freedom')
plt.ylabel('RC')
plt.legend(['double t', 't'])

plt.tight_layout()

plt.show()

print("\n\n --------------- Standard RC ---------------\n")

# Risk weights
RW_SG = 1.5  # Risk weight associated to B Corporate bonds
RW_AR = 1  # Risk weight associated to BBB Corporate bonds

# Standard Regulatory Capital
RC_SG_Std = 0.08 * RW_SG
RC_AR_Std = 0.08 * RW_AR

print(f"Standard Regulatory Capitals:         SG = {RC_SG_Std:.4f}      AR = {RC_AR_Std:.4f}")

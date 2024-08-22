import numpy as np
from scipy.stats import t, norm, kstest, ks_2samp, mannwhitneyu

def correlation_IRB(PD):
    """
    This function computes the correlation between assets chosen by the Basel Committee

    Parameters:
    PD : Probability to default

    Returns:
    R : Correlation coefficient
    """
    Rmin = 0.12
    Rmax = 0.24
    k = 50
    R = Rmin * (1 - np.exp(-k * PD)) / (1 - np.exp(-k)) + Rmax * (1 - (1 - np.exp(-k * PD)) / (1 - np.exp(-k)))
    return R

def montecarlo_RC(N, k_hat, sigma_k, LGD_cap, sigma_LGD, alpha, g, M, flag, correlation):
    """
    This function computes the Regulatory capital of a LHP using the Vasicek
    model via Monte Carlo approach

    Parameters:
    N : Number of simulations
    k_hat : Mean of k (Gaussian cumulative inverse default probability) values
    sigma_k : Standard deviation of k values
    LGD_cap : Loss Given Default mean
    sigma_LGD : Loss Given Default standard deviation
    alpha : Confidence level
    g : Standard Gaussian N-dim vector
    M : Random variable
    flag :
        1 -> Fix k = k_hat for each k
        2 -> Fix LGD as LGD_cap
        3 -> Consider LGD and k independent
        4 -> Consider LGD and k dependent
    correlation : Correlation between LGD and k, mandatory only if flag == 4

    Returns:
        EL_vec : Loss vector containing the loss for each simulation
        EL : Expected Loss
        RC : Regulatory Capital
        RC_CI : Regulatory capital confidence interval (confidence level = alpha)
        EL_CI : Expected Loss confidence interval (confidence level = alpha)
    """

    alpha_ci = 0.999

    if flag == 1:
        LGD_MC = LGD_cap + sigma_LGD * g[0, :]
        rho = correlation_IRB(norm.cdf(k_hat))
        EL_vec = LGD_MC * norm.cdf((k_hat - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)

        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        RC_CI = np.zeros((len(alpha), 2))
        EL_CI = np.zeros((len(alpha), 2))
        for i in range(len(alpha)):
            term = (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2
            RC_CI[i, 1] = EL_quantile[i] - EL + t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + term))
            RC_CI[i, 0] = EL_quantile[i] - EL - t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + term))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    elif flag == 2:
        k_MC = k_hat + sigma_k * g[1, :]
        rho = correlation_IRB(norm.cdf(k_MC))
        EL_vec = LGD_cap * norm.cdf((k_MC - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)

        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        RC_CI = np.zeros((len(alpha), 2))
        EL_CI = np.zeros((len(alpha), 2))
        for i in range(len(alpha)):
            term = (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2
            RC_CI[i, 1] = EL_quantile[i] - EL + t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + term))
            RC_CI[i, 0] = EL_quantile[i] - EL - t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + term))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    elif flag == 3:
        K_MC = k_hat + sigma_k * g[0, :]
        LGD_MC = LGD_cap + sigma_LGD * g[1, :]
        rho = correlation_IRB(norm.cdf(K_MC))

        EL_vec = LGD_MC * norm.cdf((K_MC - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)

        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        RC_CI = np.zeros((len(alpha), 2))
        EL_CI = np.zeros((len(alpha), 2))
        for i in range(len(alpha)):
            term = (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2
            RC_CI[i, 1] = EL_quantile[i] - EL + t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + term))
            RC_CI[i, 0] = EL_quantile[i] - EL - t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + term))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    elif flag == 4:
        correlation_matrix = np.array([[1, correlation], [correlation, 1]])
        A = np.linalg.cholesky(correlation_matrix)
        x = A @ g

        K_MC = k_hat + sigma_k * x[0, :]
        LGD_MC = LGD_cap + sigma_LGD * x[1, :]
        rho = correlation_IRB(norm.cdf(K_MC))

        EL_vec = LGD_MC * norm.cdf((K_MC - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)

        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        RC_CI = np.zeros((len(alpha), 2))
        EL_CI = np.zeros((len(alpha), 2))
        for i in range(len(alpha)):
            term = (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2
            RC_CI[i, 1] = EL_quantile[i] - EL + t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + term))
            RC_CI[i, 0] = EL_quantile[i] - EL - t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + term))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    return EL_vec, EL, RC, RC_CI, EL_CI

def Add_on(RC, RC_naive, EL, EL_naive, RC_CI, EL_CI):
    """
    This function computes the Add-on on the Regulatory capital after a particular stress.

    INPUTS:
    RC :            Regulatory capital
    RC_naive :      Regulatory capital naive approach
    EL :            Expected loss
    EL_naive :      Expected loss naive approach
    RC_CI :         Regulatory capital confidence interval
    EL_CI :         Expected loss confidence interval

    OUTPUTS:
    Add_on :        Regulatory capital Add-on (Rc_new = RC * (1 + Add_on))
    Add_on_CI :     Add-on confidence interval
    """

    # Compute Add-on
    Add_on = ((RC - RC_naive) + (EL - EL_naive)) / RC_naive

    # Compute Add-on confidence intervals
    if RC_CI.size > 2:
        Add_on_CI = np.zeros((len(RC_CI), 2))
        Add_on_CI[:, 0] = ((RC_CI[:, 0] - RC_naive) + (EL_CI[:, 0] - EL_naive)) / RC_naive
        Add_on_CI[:, 1] = ((RC_CI[:, 1] - RC_naive) + (EL_CI[:, 1] - EL_naive)) / RC_naive
    else:
        Add_on_CI = np.zeros((1, 2))
        Add_on_CI = ((RC_CI - RC_naive) + (EL_CI - EL_naive)) / RC_naive

    return Add_on, Add_on_CI


def montecarlo_RC_HP(N, k_hat, sigma_k, LGD_cap, sigma_LGD, alpha, g, M, epsilon, flag, N_obligors, correlation):
    """
    This function computes the Regulatory capital using a Monte Carlo
    approach in the Homogeneous portfolio case (finite Number of obligors)

    Parameters
    ----------
    N : Number of simulations
    k_hat : Mean of k (gaussian cumulative inverse default probability) values
    sigma_k : k values standard deviation
    LGD_cap : Loss Given Default mean
    sigma_LGD : Loss Given Default standard deviation
    alpha : Confidence levels
    g : Standard gaussian N-dim vector
    M : Market factors
    epsilon : Standard normal random variables
    flag :
        1 -> Fix k = k_hat
        2 -> Fix LGD = LGD_cap
        3 -> Consider LGD and k independent
        4 -> Consider LGD and k dependent
    N_obligors : Number of obligors
    correlation : Correlation matrix (mandatory for flag = 4)

    Returns
    -------
    EL_vec : Loss vector which contains the loss for each simulation
    EL : Expected Loss
    RC : Regulatory Capital
    RC_CI : Regulatory capital Confidence interval (confidence level = alpha)
    EL_CI : Expected Loss Confidence interval (confidence level = alpha)
    """

    X = np.zeros((N_obligors, N))
    alpha_ci = 0.999  # confidence level

    if flag == 1:
        k_MC = k_hat
        LGD_MC = LGD_cap + sigma_LGD * g[1, :]
        rho = correlation_IRB(norm.cdf(k_MC))

        # Risk factors
        X = np.sqrt(rho) * M + np.sqrt(1 - rho) * epsilon

        # Defaults
        Defaults = X < k_MC

        # Total Losses
        Loss = LGD_MC * Defaults

        EL_vec = np.mean(Loss, axis=0)
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        RC_CI = np.zeros((len(alpha), 2))
        EL_CI = np.zeros((len(alpha), 2))
        for i in range(len(alpha)):
            RC_CI[i, 1] = (EL_quantile[i] - EL +
                           t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                                np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                                    np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2)))
            RC_CI[i, 0] = (EL_quantile[i] - EL -
                           t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                                np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                                    np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2)))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    if flag == 2:
        k_MC = k_hat + sigma_k * g[0, :]
        LGD_MC = LGD_cap
        rho = correlation_IRB(norm.cdf(k_MC))

        # Risk factors
        X = np.sqrt(rho) * M + np.sqrt(1 - rho) * epsilon

        # Defaults
        Defaults = X < k_MC

        # Total Losses
        Loss = LGD_MC * Defaults
        EL_vec = np.mean(Loss, axis=0)
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        RC_CI = np.zeros((len(alpha), 2))
        EL_CI = np.zeros((len(alpha), 2))
        for i in range(len(alpha)):
            RC_CI[i, 1] = (EL_quantile[i] - EL +
                           t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                                np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                                    np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2)))
            RC_CI[i, 0] = (EL_quantile[i] - EL -
                           t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                                np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                                    np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2)))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    if flag == 3:
        k_MC = k_hat + sigma_k * g[0, :]
        LGD_MC = LGD_cap + sigma_LGD * g[1, :]
        rho = correlation_IRB(norm.cdf(k_MC))

        # Risk factors
        X = np.sqrt(rho) * M + np.sqrt(1 - rho) * epsilon

        # Defaults
        Defaults = X < k_MC

        # Total Losses
        Loss = LGD_MC * Defaults

        EL_vec = np.mean(Loss, axis=0)
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        RC_CI = np.zeros((len(alpha), 2))
        EL_CI = np.zeros((len(alpha), 2))
        for i in range(len(alpha)):
            RC_CI[i, 1] = (EL_quantile[i] - EL +
                           t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                                np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                                    np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2)))
            RC_CI[i, 0] = (EL_quantile[i] - EL -
                           t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                                np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                                    np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2)))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    if flag == 4:
        correlation_matrix = np.array([[1, correlation], [correlation, 1]])
        g = np.linalg.cholesky(correlation_matrix) @ g
        k_MC = k_hat + sigma_k * g[0, :]
        LGD_MC = LGD_cap + sigma_LGD * g[1, :]

        rho = correlation_IRB(norm.cdf(k_MC))

        # Risk factors
        X = np.sqrt(rho) * M + np.sqrt(1 - rho) * epsilon

        # Defaults
        Defaults = X < k_MC

        # Total Losses
        Loss = LGD_MC * Defaults

        EL_vec = np.mean(Loss, axis=0)
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        RC_CI = np.zeros((len(alpha), 2))
        EL_CI = np.zeros((len(alpha), 2))
        for i in range(len(alpha)):
            RC_CI[i, 1] = (EL_quantile[i] - EL +
                           t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                                np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                                    np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2)))
            RC_CI[i, 0] = (EL_quantile[i] - EL -
                           t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                                np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                                    np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2)))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    return EL_vec, EL, RC, RC_CI, EL_CI

def montecarlo_RC_T_double(N, k_hat, sigma_k, LGD_cap, sigma_LGD, alpha, t1, t2, M, flag, correlation):
    """
    Computes the Regulatory Capital using a Monte Carlo approach, simulating k and LGD as double t-distributions.

    Parameters:
    N: Number of simulations
    k_hat: Mean of k (Gaussian cumulative inverse default probability) values
    sigma_k: Standard deviation of k values
    LGD_cap: Mean Loss Given Default
    sigma_LGD: Standard deviation of Loss Given Default
    alpha: Confidence level
    t1: 2 x N matrix of t-distribution values for LGD
    t2: 2 x N matrix of t-distribution values for k
    flag: Determines the simulation approach:
          1 -> Fix k = k_hat for each k
          2 -> Fix LGD = LGD_cap
          3 -> Consider LGD and k as independent
          4 -> Consider LGD and k as dependent
    correlation: Correlation between LGD and k, required if flag==4 (default is None)

    Returns:
    tuple: A tuple containing:
        - EL_vec: Loss vector containing the loss for each simulation
        - EL: Expected Loss
        - RC: Regulatory Capital
        - RC_CI: Regulatory Capital Confidence Interval (confidence level = alpha)
        - EL_CI: Expected Loss Confidence Interval (confidence level = alpha)
    """

    rho_v = 0.5  # correlation double t-student
    alpha_ci = 0.999  # confidence level
    EL_vec = np.zeros(N)
    RC_CI = np.zeros((len(alpha), 2))
    EL_CI = np.zeros((len(alpha), 2))

    if flag == 1:
        LGD_MC = LGD_cap + sigma_LGD * (np.sqrt(rho_v) * t1[0] + np.sqrt(1 - rho_v) * t1[1])
        rho = correlation_IRB(norm.cdf(k_hat))
        EL_vec = LGD_MC * norm.cdf((k_hat - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        for i in range(len(alpha)):
            RC_CI[i, 0] = EL_quantile[i] - EL + t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile[i] - EL - t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N), EL - t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    if flag == 2:
        k_MC = k_hat + sigma_k * (np.sqrt(rho_v) * t2[0] + np.sqrt(1 - rho_v) * t2[1])
        rho = correlation_IRB(norm.cdf(k_MC))
        EL_vec = LGD_cap * norm.cdf((k_MC - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        for i in range(len(alpha)):
            RC_CI[i, 0] = EL_quantile[i] - EL + t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile[i] - EL - t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N), EL - t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    if flag == 3:
        LGD_MC = LGD_cap + sigma_LGD * (np.sqrt(rho_v) * t1[0] + np.sqrt(1 - rho_v) * t1[1])
        k_MC = k_hat + sigma_k * (np.sqrt(rho_v) * t2[0] + np.sqrt(1 - rho_v) * t2[1])
        rho = correlation_IRB(norm.cdf(k_MC))
        EL_vec = LGD_MC * norm.cdf((k_MC - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        for i in range(len(alpha)):
            RC_CI[i, 0] = EL_quantile[i] - EL + t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile[i] - EL - t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N), EL - t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    if flag == 4:
        sigma = np.array([[1, correlation], [correlation, 1]])
        A = np.linalg.cholesky(sigma).T
        x = np.dot(A, np.vstack([(np.sqrt(rho_v) * t1[0, :] + np.sqrt(1 - rho_v) * t1[1, :]),
                                 (np.sqrt(rho_v) * t2[0, :] + np.sqrt(1 - rho_v) * t2[1, :])]))
        K_MC = k_hat + sigma_k * x[1, :]
        LGD_MC = LGD_cap + sigma_LGD * x[0, :]
        rho = correlation_IRB(norm.cdf(K_MC))
        EL_vec = LGD_MC * norm.cdf((K_MC - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        for i in range(len(alpha)):
            RC_CI[i, 0] = EL_quantile[i] - EL + t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (
                        np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                            np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile[i] - EL - t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (
                        np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                            np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    return EL_vec, EL, RC, RC_CI, EL_CI

def montecarlo_RC_HP_T_double(N, k_hat, sigma_k, LGD_cap, sigma_LGD, alpha, t1, t2, M, epsilon, flag, N_obligors, correlation):
    """
    This function computes the Regulatory Capital using a Monte Carlo approach in the homogeneous portfolio case
    (finite number of obligors), considering LGD and k simulated as a double t-distribution.

    Parameters:
    N: Number of simulations
    k_hat: Mean of k (Gaussian cumulative inverse default probability) values
    sigma_k: Standard deviation of k values
    LGD_cap: Mean Loss Given Default
    sigma_LGD: Standard deviation of Loss Given Default
    alpha: Confidence level
    g: Standard Gaussian N-dimensional vector
    flag: Determines the simulation approach:
          1 -> Fix k = k_hat
          2 -> Fix LGD = LGD_cap
          3 -> Consider LGD and k as independent
          4 -> Consider LGD and k as dependent
    N_obligors: Number of obligors
    correlation: Correlation matrix, required if flag==4 (default is None)

    Returns:
    tuple: A tuple containing:
        - EL_vec: Loss vector containing the loss for each simulation
        - EL: Expected Loss
        - RC: Regulatory Capital
        - RC_CI: Regulatory Capital Confidence Interval (confidence level = alpha)
        - EL_CI: Expected Loss Confidence Interval (confidence level = alpha)
    """

    X = np.zeros((N_obligors, N))
    alpha_ci = 0.999  # confidence level
    rho_v = 0.5  # correlation double t-student
    EL_vec = np.zeros(N)
    EL = 0
    RC = 0
    RC_CI = np.zeros((len(alpha), 2))
    EL_CI = np.zeros((len(alpha), 2))

    if flag == 1:
        k_MC = k_hat
        LGD_MC = LGD_cap + sigma_LGD * (np.sqrt(rho_v) * t1[0, :] + np.sqrt(1 - rho_v) * t1[1, :])
        rho = correlation_IRB(norm.cdf(k_MC))

        # Risk factors
        X = np.sqrt(rho) * M + np.sqrt(1 - rho) * epsilon

        # Defaults
        Defaults = X < k_MC

        # Total Losses
        Loss = LGD_MC * Defaults

        EL_vec = np.mean(Loss, axis=0)
        EL = np.mean(EL_vec)

        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        for i in range(len(alpha)):
            RC_CI[i, 0] = EL_quantile[i] - EL + t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile[i] - EL - t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N), EL - t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    if flag == 2:
        k_MC = k_hat + sigma_k * (np.sqrt(rho_v) * t2[0, :] + np.sqrt(1 - rho_v) * t2[1, :])
        rho = correlation_IRB(norm.cdf(k_MC))
        LGD_MC = LGD_cap

        # Risk factors
        X = np.sqrt(rho) * M + np.sqrt(1 - rho) * epsilon

        # Defaults
        Defaults = X < k_MC

        # Total Losses
        Loss = LGD_MC * Defaults
        EL_vec = np.mean(Loss, axis=0)
        EL = np.mean(EL_vec)

        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        # Confidence interval computation
        for i in range(len(alpha)):
            RC_CI[i, 0] = EL_quantile[i] - EL + t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile[i] - EL - t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N), EL - t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    if flag == 3:
        LGD_MC = LGD_cap + sigma_LGD * (np.sqrt(rho_v) * t1[0, :] + np.sqrt(1 - rho_v) * t1[1, :])
        k_MC = k_hat + sigma_k * (np.sqrt(rho_v) * t2[0, :] + np.sqrt(1 - rho_v) * t2[1, :])
        rho = correlation_IRB(norm.cdf(k_MC))

        # Risk factors
        X = np.sqrt(rho) * M + np.sqrt(1 - rho) * epsilon

        # Defaults
        Defaults = X < k_MC

        # Total Losses
        Loss = LGD_MC * Defaults

        EL_vec = np.mean(Loss, axis=0)
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)

        RC = EL_quantile - EL

        # Confidence interval computation
        for i in range(len(alpha)):
            RC_CI[i, 0] = EL_quantile[i] - EL + t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile[i] - EL - t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N), EL - t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    if flag == 4:
        sigma = np.array([[1, correlation], [correlation, 1]])
        A = np.linalg.cholesky(sigma).T
        x = np.dot(A, np.vstack([(np.sqrt(rho_v) * t1[0, :] + np.sqrt(1 - rho_v) * t1[1, :]),
                                 (np.sqrt(rho_v) * t2[0, :] + np.sqrt(1 - rho_v) * t2[1, :])]))
        K_MC = k_hat + sigma_k * x[1, :]
        LGD_MC = LGD_cap + sigma_LGD * x[0, :]
        rho = correlation_IRB(norm.cdf(K_MC))

        # Risk factors
        X = np.sqrt(rho) * M + np.sqrt(1 - rho) * epsilon

        # Defaults
        Defaults = X < K_MC

        # Total Losses
        Loss = LGD_MC * Defaults

        EL_vec = np.mean(Loss, axis=0)
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)

        RC = EL_quantile - EL

        # Confidence interval computation
        for i in range(len(alpha)):
            RC_CI[i, 0] = EL_quantile[i] - EL + t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (
                        np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                            np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile[i] - EL - t.ppf((1 - alpha_ci) / 2, N) * np.sqrt(2 / N * (
                    np.var(EL_vec) + (alpha[i] * (1 - alpha[i])) / (
                        np.exp(-EL_quantile[i] ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                       EL - t.ppf((1 - alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    return EL_vec, EL, RC, RC_CI, EL_CI

def montecarlo_RC_T(N, k_hat, sigma_k, LGD_cap, sigma_LGD, alpha, t_samples, M, flag, correlation=None):
    """
        This function computes the Regulatory capital using a Montecarlo approach
        simulating k and LGD as t-student

        INPUTS
        N :           Number of simulations
        k_hat :       Mean of k (gaussian cumulative inverse default probability) values
        sigma_k :     k values standard deviation
        LGD_cap :     Loss Given Default mean
        sigma_LGD :   Loss Given Default standard deviation
        alpha :       Confidence level
        t :           std gaussian N-dim vector
        flag :        1 -> Fix k = k_hat for each k
                      2 -> Fix LGD as LGD_cap
                      3 -> Consider LGD and k independent
                      4 -> Consider LGD and k dependent
        correlation : Correlation between LGD and k, mandatory only if flag==4

        OUTPUTS
        EL_vec :      Loss vector which contains the loss for each simulation
        EL :          Expected Loss
        RC :          Regulatory Capital
        RC_CI :       Regulatory capital Confidence interval (confidence level = alpha)
        EL_CI :       Expected Loss Confidence interval (confidence level = alpha)
        """

    alpha_ci = 0.999  # confidence level

    if flag == 1:
        LGD_MC = LGD_cap + sigma_LGD * t_samples[0, :]
        rho = correlation_IRB(norm.cdf(k_hat))

        EL_vec = LGD_MC * norm.cdf((k_hat - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        RC_CI = np.zeros((1, 2))
        EL_CI = np.zeros((1, 2))
        for i in range(1):
            RC_CI[i, 0] = EL_quantile - EL + t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha * (1 - alpha)) / (np.exp(-EL_quantile**2 / 2) / np.sqrt(2 * np.pi))**2))
            RC_CI[i, 1] = EL_quantile - EL - t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (np.var(EL_vec) + (alpha * (1 - alpha)) / (np.exp(-EL_quantile**2 / 2) / np.sqrt(2 * np.pi))**2))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N), EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    elif flag == 2:
        k_MC = k_hat + sigma_k * t_samples[1, :]
        rho = correlation_IRB(norm.cdf(k_MC))

        EL_vec = LGD_cap * norm.cdf((k_MC - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        RC_CI = np.zeros((1, 2))
        EL_CI = np.zeros((1, 2))
        for i in range(1):
            RC_CI[i, 0] = EL_quantile - EL + t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                        np.var(EL_vec) + (alpha * (1 - alpha)) / (
                            np.exp(-EL_quantile ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile - EL - t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                        np.var(EL_vec) + (alpha * (1 - alpha)) / (
                            np.exp(-EL_quantile ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    elif flag == 3:
        k_MC = k_hat + sigma_k * t_samples[0, :]
        LGD_MC = LGD_cap + sigma_LGD * t_samples[1, :]
        rho = correlation_IRB(norm.cdf(k_MC))

        EL_vec = LGD_MC * norm.cdf((k_MC - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        RC_CI = np.zeros((1, 2))
        EL_CI = np.zeros((1, 2))
        for i in range(1):
            RC_CI[i, 0] = EL_quantile - EL + t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                        np.var(EL_vec) + (alpha * (1 - alpha)) / (
                            np.exp(-EL_quantile ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile - EL - t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                        np.var(EL_vec) + (alpha * (1 - alpha)) / (
                            np.exp(-EL_quantile ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    elif flag == 4:
        sigma = np.array([[1, correlation], [correlation, 1]])
        A = np.linalg.cholesky(sigma).T
        x = np.dot(A, t_samples)

        LGD_MC = LGD_cap + sigma_LGD * x[0, :]
        k_MC = k_hat + sigma_k * x[1, :]
        rho = correlation_IRB(norm.cdf(k_MC))

        EL_vec = LGD_MC * norm.cdf((k_MC - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        EL = np.mean(EL_vec)
        EL_quantile = np.quantile(EL_vec, alpha)
        RC = EL_quantile - EL

        RC_CI = np.zeros((1, 2))
        EL_CI = np.zeros((1, 2))
        for i in range(1):
            RC_CI[i, 0] = EL_quantile - EL + t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                        np.var(EL_vec) + (alpha * (1 - alpha)) / (
                            np.exp(-EL_quantile ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            RC_CI[i, 1] = EL_quantile - EL - t.ppf((1 + alpha_ci) / 2, N) * np.sqrt(2 / N * (
                        np.var(EL_vec) + (alpha * (1 - alpha)) / (
                            np.exp(-EL_quantile ** 2 / 2) / np.sqrt(2 * np.pi)) ** 2))
            EL_CI[i, :] = [EL + t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N),
                           EL - t.ppf((1 + alpha_ci) / 2, N) * np.std(EL_vec) / np.sqrt(N)]

    return EL_vec, EL, RC, RC_CI, EL_CI
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import lognorm


def uniform_1(low, high, size):
    return np.random.uniform(low, high, size)

def normal_2(low, high, size):
    mean_x = (high + low) / 2
    sd_x = (high - low) / 4
    return np.random.normal(mean_x, sd_x, size)

def normal_3(low, central, high, size):
    mean_x = central
    sd_x = (high - low) / 4
    return np.random.normal(mean_x, sd_x, size)

def log_normal_4(low, central, high, size):
    def f(sigma):
        mu = np.log(central) + sigma**2
        return np.abs(
            lognorm.cdf(high, sigma, scale=np.exp(mu))
            - lognorm.cdf(low, sigma, scale=np.exp(mu))
            - 0.95
        )

    sigma_x = minimize_scalar(f, bounds=(0, 1), method="bounded").x
    mu_x = np.log(central) + sigma_x**2
    return lognorm.rvs(sigma_x, scale=np.exp(mu_x), size=size)

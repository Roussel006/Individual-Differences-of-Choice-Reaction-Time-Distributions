"""
This code contains all functions necessary to investigate RT distributions


1. All functions to calculate Entropy, RE, and OI for different distributions (gamma, exg)
    Some functions are numerical (e.g., all for exg)
2. Fitting RT to gamma and exg distributions.
3. Fitting RT to DDM
    To deal with the stochasticity, we do bootstrapping.
"""


import numpy as np
# import pymc3 as pm
# import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
# To fit gamma dists
import scipy.stats as stats
# from scipy.special import gamma ## DID NOT USE THIS, BUT HELPS TO REMEMBER that i did not use it
from scipy.special import gamma, gammaln, digamma, erf, erfc
from scipy.stats import norm
import scipy.integrate
import time
import copy

import pyddm
# OLD
# from pyddm import Model, Fittable
# from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture, ICPointSourceCenter
# from pyddm.functions import fit_adjust_model, display_model
# from pyddm import Sample

import ipywidgets as widgets
import cv2

import time
from collections import Counter, OrderedDict
import itertools
from itertools import chain
import copy
import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='once')

# ------------ Vector functions -----------------------------

def unit_vector(A):
     magnitude = np.sqrt(np.sum(A**2))
     u = np.array(A)/magnitude
     return(u)

def cosine_similarity_RR(A, B):
     A = np.array(A)
     B = np.array(B)
     uA = unit_vector(A)
     uB = unit_vector(B)
     cs = np.sum(uA * uB)
     return(cs)

def euclid_distance_RR(A, B):
     A = np.array(A)
     B = np.array(B)
     d = np.sqrt(np.sum((A-B)**2))
     return(d)

## ------------- NORMAL DISTRIBUTION THINGS --------------------
## ------------------------------------------------------------

"""
Some important notes:

     (1) Distributions: (a) 2-parameter Gamma Distribution, similar to Torres et al. (2017) 
                    and (b) 3-parameter Ex-Gaussian Distribution (AKA exponential distribution)
     
     (2) Distributional similarity and distances:
          (a) Relative Entropy for distance or difference between distributions in terms of bits of information.
          (b) Bhattacharrya Co-efficient, a statistical measure of distributional similarity
          (c) Overlapping index, measuring the overlapping area between two distributions.

     Closed form solutions for the integrals in these measures can be quite complicated, especially for Ex-Gaussian Distributions. But these integrals can be easily and accurately performed numerically in multiple ways. Pastore et al suggested a kernel-based method that does not require assumption of distribution type.
     First, we reduce the infinite integral to a definite integral. For each pair of distributions, we calculate the shortest range that contains at least 99.98% of each distribution.
     Then, we numerically integrate using the quadrature method.

     Another possible distributional distance that we do not explore here: Wasserstein distance.

     (3) We examine the four moments characterize the distributions. These moments do not require assumption of distribution shape. 

"""

## ------------- GAMMA DISTRIBUTION THINGS --------------------
## ------------------------------------------------------------

# (1) Entropy of Gamma distributions
def func_gamma_scipy_Entropy_integrand(x, params_px):

     k_x, theta_x = params_px
     # k_ref, theta_ref = params_pref

     px = stats.gamma.pdf(x, a = k_x, loc = 0, scale = theta_x)
     # pref = stats.gamma.pdf(x, a = k_ref, loc = 0, scale = theta_ref)

     # return(px)
     integrand = px*np.log2(px)
     # integrand = px

     return(integrand)

def Entropy_gamma_numerical(params_px):
# Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, theta_x = params_px

     lower_lim = stats.gamma.ppf(0.0001, a = k_x, loc = 0, scale = theta_x)
     upper_lim = stats.gamma.ppf(0.9999, a = k_x, loc = 0, scale = theta_x)

     # params_px = np.array([k_x, mu_x, sigma_x])
     # params_pref = np.array([k_ref, mu_ref, sigma_ref])

     entropy_gamma_num = -1* scipy.integrate.quad(func_gamma_scipy_Entropy_integrand, lower_lim, upper_lim, args = list(params_px))[0]
     # entropy_exg_scipy_num = scipy.integrate.quad(func_exg_scipy_Entropy_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(entropy_gamma_num)

def Entropy_gamma_analytical(params_px):
# Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, theta_x = params_px

     # lower_lim = stats.gamma.ppf(0.0001, a = k_x, loc = 0, scale = theta_x)
     # upper_lim = stats.gamma.ppf(0.9999, a = k_x, loc = 0, scale = theta_x)

     # params_px = np.array([k_x, mu_x, sigma_x])
     # params_pref = np.array([k_ref, mu_ref, sigma_ref])

     entropy_gamma_ana = k_x + np.log(theta_x) + gammaln(k_x) + (1-k_x)*digamma(k_x)
     entropy_gamma_ana = entropy_gamma_ana/np.log(2) # Convert "nats" to "bits"
     # entropy_gamma_scipy_num = scipy.integrate.quad(func_gamma_scipy_Entropy_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(entropy_gamma_ana)


# RE function

## (2a) RE, analytical

def RE_uni_gamma_analytical(params_px, params_pref):

     k_x, theta_x = params_px
     k_ref, theta_ref = params_pref

     RE = gammaln(k_ref) - gammaln(k_x) + \
               (k_x - k_ref) * digamma(k_x) + \
               k_ref * np.log(theta_ref/theta_x) + \
               k_x * (theta_x - theta_ref) / theta_ref # In nats
     ## Ideally, I would liked to use np.log(gamma(.))
     ## But it goes outta bound easily with moderate arguments to gammma function
     ## So, I use gammaln which takes care of the issue

     RE = RE/ np.log(2) # In bits
     return RE


##  -------> (2b) RE, numerical
     # We do it in two steps: (1) the integrand function, and (2) the integral using it.
     # NOTE: When integrating we use the range corresponding to 0.0001 and 0.9999 of their distribution.

def func_gamma_RE_integrand(x, params_px, params_pref): # v1

     k_x, theta_x = params_px
     k_ref, theta_ref = params_pref

     px = 1/gamma(k_x)/theta_x**k_x * x**(k_x - 1) * np.exp(-x/theta_x)
     pref = 1/gamma(k_ref)/theta_ref**k_ref * x**(k_ref - 1) * np.exp(-x/theta_ref)

  
     integrand = px*np.log2(px) - px * np.log2(pref) # an f(x)
     return(integrand)


def RE_uni_gamma_numerical(params_px, params_pref):
     # Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, theta_x = params_px
     k_ref, theta_ref = params_pref

     # in stats, gamma function takes in a (for alpha), loc, and scale. a is our k and scale is our theta. We set loc = 0
     ppf_0001 = [stats.gamma.ppf(0.0001, a = k_ref, loc = 0, scale = theta_ref), \
                    stats.gamma.ppf(0.0001, a = k_x, loc = 0, scale = theta_x)]
     lower_lim = np.min(ppf_0001) # lower of the two ppf(0.001) points



     ppf_9999 = [stats.gamma.ppf(0.9999, a = k_ref, loc = 0, scale = theta_ref), \
                    stats.gamma.ppf(0.9999, a = k_x, loc = 0, scale = theta_x)]
     upper_lim = np.max(ppf_9999) # higher of the two ppf(0.999) points

     re_gamma_numerical = scipy.integrate.quad(func_gamma_RE_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(re_gamma_numerical)


## (3) Bhattacharyya Coeff, numerical
     # We do it in two steps: (1) the integrand function, and (2) the integral using it.
     # NOTE: When integrating we use the range corresponding to 0.0001 and 0.9999 of their distribution.

def func_gamma_BCoeff_integrand(x, params_px, params_pref): # v1

     k_x, theta_x = params_px
     k_ref, theta_ref = params_pref

     px = 1/gamma(k_x)/theta_x**k_x * x**(k_x - 1) * np.exp(-x/theta_x)
     pref = 1/gamma(k_ref)/theta_ref**k_ref * x**(k_ref - 1) * np.exp(-x/theta_ref)

  
     integrand = np.sqrt(px*pref)
     return(integrand)


def BCoeff_uni_gamma_numerical(params_px, params_pref):
     # Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, theta_x = params_px
     k_ref, theta_ref = params_pref

     # in stats, gamma function takes in a (for alpha), loc, and scale. a is our k and scale is our theta. We set loc = 0
     ppf_0001 = [stats.gamma.ppf(0.0001, a = k_ref, loc = 0, scale = theta_ref), \
                    stats.gamma.ppf(0.0001, a = k_x, loc = 0, scale = theta_x)]
     lower_lim = np.min(ppf_0001) # lower of the two ppf(0.001) points



     ppf_9999 = [stats.gamma.ppf(0.9999, a = k_ref, loc = 0, scale = theta_ref), \
                    stats.gamma.ppf(0.9999, a = k_x, loc = 0, scale = theta_x)]
     upper_lim = np.max(ppf_9999) # higher of the two ppf(0.999) points

     bcoeff_gamma_numerical = scipy.integrate.quad(func_gamma_BCoeff_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(bcoeff_gamma_numerical)


## (4) Overlapping Index, numerical
     # We do it in two steps: (1) the integrand function, and (2) the integral using it.
     # NOTE: When integrating we use the range corresponding to 0.0001 and 0.9999 of their distribution.

def func_gamma_OI_integrand(x, params_px, params_pref): # v1

     k_x, theta_x = params_px
     k_ref, theta_ref = params_pref

     px = 1/gamma(k_x)/theta_x**k_x * x**(k_x - 1) * np.exp(-x/theta_x)
     pref = 1/gamma(k_ref)/theta_ref**k_ref * x**(k_ref - 1) * np.exp(-x/theta_ref)

  
     integrand = np.abs(px - pref)
     return(integrand)


def OI_uni_gamma_numerical(params_px, params_pref):
     # Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, theta_x = params_px
     k_ref, theta_ref = params_pref

     # in stats, gamma function takes in a (for alpha), loc, and scale. a is our k and scale is our theta. We set loc = 0
     ppf_0001 = [stats.gamma.ppf(0.0001, a = k_ref, loc = 0, scale = theta_ref), \
                    stats.gamma.ppf(0.0001, a = k_x, loc = 0, scale = theta_x)]
     lower_lim = np.min(ppf_0001) # lower of the two ppf(0.001) points


     ppf_9999 = [stats.gamma.ppf(0.9999, a = k_ref, loc = 0, scale = theta_ref), \
                    stats.gamma.ppf(0.9999, a = k_x, loc = 0, scale = theta_x)]
     upper_lim = np.max(ppf_9999) # higher of the two ppf(0.999) points

     oi_gamma_numerical = 1- 0.5*scipy.integrate.quad(func_gamma_OI_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(oi_gamma_numerical)

def func_gamma_OI_integrand2(x, params_px, params_pref): # v1

     k_x, theta_x = params_px
     k_ref, theta_ref = params_pref

     px = 1/gamma(k_x)/theta_x**k_x * x**(k_x - 1) * np.exp(-x/theta_x)
     pref = 1/gamma(k_ref)/theta_ref**k_ref * x**(k_ref - 1) * np.exp(-x/theta_ref)

  
     integrand = min(px, pref)
     return(integrand)


def OI_uni_gamma_numerical2(params_px, params_pref):
     # Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, theta_x = params_px
     k_ref, theta_ref = params_pref

     # in stats, gamma function takes in a (for alpha), loc, and scale. a is our k and scale is our theta. We set loc = 0
     ppf_0001 = [stats.gamma.ppf(0.0001, a = k_ref, loc = 0, scale = theta_ref), \
                    stats.gamma.ppf(0.0001, a = k_x, loc = 0, scale = theta_x)]
     lower_lim = np.min(ppf_0001) # lower of the two ppf(0.001) points



     ppf_9999 = [stats.gamma.ppf(0.9999, a = k_ref, loc = 0, scale = theta_ref), \
                    stats.gamma.ppf(0.9999, a = k_x, loc = 0, scale = theta_x)]
     upper_lim = np.max(ppf_9999) # higher of the two ppf(0.999) points

     oi_gamma_numerical = scipy.integrate.quad(func_gamma_OI_integrand2, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(oi_gamma_numerical)


# ------------ Ex-Gaussian Distribution Things ----------------------
## ------------------------------------------------------------

# (1) Entropy of ex-Gaussian distibutions
def func_exg_scipy_Entropy_integrand(x, params_px):

     k_x, mu_x, sigma_x = params_px
     # k_ref, mu_ref, sigma_ref = params_pref

     px = stats.exponnorm.pdf(x, K = k_x, loc = mu_x, scale = sigma_x)
     # pref = stats.exponnorm.pdf(x, K = k_ref, loc = mu_ref, scale = sigma_ref)

     # return(px)
     integrand = px*np.log2(px)
     # integrand = px

     return(integrand)

def Entropy_exg_numerical(params_px):
# Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, mu_x, sigma_x = params_px

     lower_lim = stats.exponnorm.ppf(0.0001, K = k_x, loc = mu_x, scale = sigma_x)
     upper_lim = stats.exponnorm.ppf(0.9999, K = k_x, loc = mu_x, scale = sigma_x)

     # params_px = np.array([k_x, mu_x, sigma_x])
     # params_pref = np.array([k_ref, mu_ref, sigma_ref])

     entropy_exg_num = -1* scipy.integrate.quad(func_exg_scipy_Entropy_integrand, lower_lim, upper_lim, args = list(params_px))[0]
     # entropy_exg_scipy_num = scipy.integrate.quad(func_exg_scipy_Entropy_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(entropy_exg_num)


## -------> (2) RE, numerical

# NOTE: WE DO NOT KNOW THE ANALYTICAL FORM.
     # We do it in two steps: (1) the integrand function, and (2) the integral using it.
     # NOTE: When integrating we use the range corresponding to 0.0001 and 0.9999 of their distribution.

# Note: We did the integration numerically, with a clever trick to deal with the infinite limits.
# v1 and v2 checked to ensure they yield the same answer

def func_exg_RE_integrand(x, params_px, params_pref): # v1

     k_x, mu_x, sigma_x = params_px
     k_ref, mu_ref, sigma_ref = params_pref

     px = 1/(2*sigma_x*k_x) * np.exp(1/ (2*k_x**2) - (x - mu_x) / (sigma_x*k_x)) * \
                                        erfc(-1*( ( (x-mu_x)/sigma_x -1/k_x)/np.sqrt(2) ) )
     pref = 1/(2*sigma_ref*k_ref) * np.exp(1/ (2*k_ref**2) - (x - mu_ref) / (sigma_ref*k_ref)) * \
                                        erfc(-1*( ( (x-mu_ref)/sigma_ref -1/k_ref)/np.sqrt(2) ) )
     
     integrand = px*np.log2(px) - px * np.log2(pref) # an f(x)
     return(integrand)

# def func_exg_scipy_RE_integrand(x, params_px, params_pref): # v2

#      k, mu, sigma = params_px
#      k_ref, mu_ref, sigma_ref = params_pref

#      px = stats.exponnorm.pdf(x, K = k, loc = mu, scale = sigma)
#      pref = stats.exponnorm.pdf(x, K = k_ref, loc = mu_ref, scale = sigma_ref)

#      integrand = px*np.log2(px) - px * np.log2(pref) # an f(x)
#      return(integrand)

def RE_uni_exg_numerical(params_px, params_pref):
# Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, mu_x, sigma_x = params_px
     k_ref, mu_ref, sigma_ref = params_pref     

     ppf_0001 = [stats.exponnorm.ppf(0.0001, K = k_ref, loc = mu_ref, scale = sigma_ref), \
                    stats.exponnorm.ppf(0.0001, K = k_x, loc = mu_x, scale = sigma_x)]
     lower_lim = np.min(ppf_0001) # lower of the two ppf(0.001) points

     ppf_9999 = [stats.exponnorm.ppf(0.9999, K = k_ref, loc = mu_ref, scale = sigma_ref), \
                    stats.exponnorm.ppf(0.9999, K = k_x, loc = mu_x, scale = sigma_x)]
     upper_lim = np.max(ppf_9999) # higher of the two ppf(0.999) points

     # params_px = np.array([k_x, mu_x, sigma_x])
     # params_pref = np.array([k_ref, mu_ref, sigma_ref])

     re_exg_num = scipy.integrate.quad(func_exg_RE_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     # re_exg_scipy_num = scipy.integrate.quad(func_exg_scipy_RE_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(re_exg_num)


def calculate_exg_params_and_then_RE(list_of_RT_arrays_to_compare):
     num_arrays = len(list_of_RT_arrays_to_compare)
     RE_matrix = np.zeros([num_arrays, num_arrays])

     for array_i in range(num_arrays):
          for array_j in range(num_arrays):
               array_px = list_of_RT_arrays_to_compare[array_i]
               array_pref = list_of_RT_arrays_to_compare[array_j]
               exg_params_px = stats.exponnorm.fit(array_px)
               exg_params_pref = stats.exponnorm.fit(array_pref)
               RE_matrix[array_i, array_j] = RE_uni_exg(exg_params_px, exg_params_pref)
               print(exg_params_px, exg_params_pref, RE_matrix)

     return(RE_matrix)


# ------------------> (3) BCoeff of exponnorm distribution ---------------------------------
# Note: We did the integration numerically, with a clever trick to deal with the infinite limits.
# v1 and v2 to be checked to ensure they yield the same answer

def func_exg_BCoeff_integrand(x, params_px, params_pref): # v1

     k_x, mu_x, sigma_x = params_px
     k_ref, mu_ref, sigma_ref = params_pref

     px = 1/(2*sigma_x*k_x) * np.exp(1/ (2*k_x**2) - (x - mu_x) / (sigma_x*k_x)) * \
                                        erfc(-1*( ( (x-mu_x)/sigma_x -1/k_x )/np.sqrt(2) ) )
     pref = 1/(2*sigma_ref*k_ref) * np.exp(1/ (2*k_ref**2) - (x - mu_ref) / (sigma_ref*k_ref)) * \
                                        erfc(-1*( ( (x-mu_ref)/sigma_ref -1/k_ref)/np.sqrt(2) ) )
     
     integrand = np.sqrt(px*pref) # an f(x)
     return(integrand)


def BCoeff_uni_exg_numerical(params_px, params_pref): # Bhattacharya distance
     # Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, mu_x, sigma_x = params_px
     k_ref, mu_ref, sigma_ref = params_pref     

     ppf_0001 = [stats.exponnorm.ppf(0.0001, K = k_ref, loc = mu_ref, scale = sigma_ref), \
                    stats.exponnorm.ppf(0.0001, K = k_x, loc = mu_x, scale = sigma_x)]
     lower_lim = np.min(ppf_0001) # lower of the two ppf(0.001) points

     ppf_9999 = [stats.exponnorm.ppf(0.9999, K = k_ref, loc = mu_ref, scale = sigma_ref), \
                    stats.exponnorm.ppf(0.9999, K = k_x, loc = mu_x, scale = sigma_x)]
     upper_lim = np.max(ppf_9999) # higher of the two ppf(0.999) points

     # params_px = np.array([k_x, mu_x, sigma_x])
     # params_pref = np.array([k_ref, mu_ref, sigma_ref])

     bcoeff_exg_num = scipy.integrate.quad(func_exg_BCoeff_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     # bdis_exg_num = -np.log2(bcoeff_exg_num)
     # re_exg_scipy_num = scipy.integrate.quad(func_exg_scipy_RE_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(bcoeff_exg_num)

## (4) Overlapping Index, numerical
     # We do it in two steps: (1) the integrand function, and (2) the integral using it.
     # NOTE: When integrating we use the range corresponding to 0.0001 and 0.9999 of their distribution.

def func_exg_OI_integrand(x, params_px, params_pref): # v1

     k_x, mu_x, sigma_x = params_px
     k_ref, mu_ref, sigma_ref = params_pref

     px = 1/(2*sigma_x*k_x) * np.exp(1/ (2*k_x**2) - (x - mu_x) / (sigma_x*k_x)) * \
                                        erfc(-1*( ( (x-mu_x)/sigma_x -1/k_x)/np.sqrt(2) ) )
     pref = 1/(2*sigma_ref*k_ref) * np.exp(1/ (2*k_ref**2) - (x - mu_ref) / (sigma_ref*k_ref)) * \
                                        erfc(-1*( ( (x-mu_ref)/sigma_ref -1/k_ref)/np.sqrt(2) ) )  
     integrand = np.abs(px - pref)
     return(integrand)


def OI_uni_exg_numerical(params_px, params_pref):
     # Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, mu_x, sigma_x = params_px
     k_ref, mu_ref, sigma_ref = params_pref

     ppf_0001 = [stats.exponnorm.ppf(0.0001, K = k_ref, loc = mu_ref, scale = sigma_ref), \
                    stats.exponnorm.ppf(0.0001, K = k_x, loc = mu_x, scale = sigma_x)]
     lower_lim = np.min(ppf_0001) # lower of the two ppf(0.001) points

     ppf_9999 = [stats.exponnorm.ppf(0.9999, K = k_ref, loc = mu_ref, scale = sigma_ref), \
                    stats.exponnorm.ppf(0.9999, K = k_x, loc = mu_x, scale = sigma_x)]
     upper_lim = np.max(ppf_9999) # higher of the two ppf(0.999) points

     oi_exg_numerical = 1- 0.5*scipy.integrate.quad(func_exg_OI_integrand, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(oi_exg_numerical)

def func_exg_OI_integrand2(x, params_px, params_pref): # v1

     k_x, mu_x, sigma_x = params_px
     k_ref, mu_ref, sigma_ref = params_pref

     px = 1/(2*sigma_x*k_x) * np.exp(1/ (2*k_x**2) - (x - mu_x) / (sigma_x*k_x)) * \
                                        erfc(-1*( ( (x-mu_x)/sigma_x -1/k_x)/np.sqrt(2) ) )
     pref = 1/(2*sigma_ref*k_ref) * np.exp(1/ (2*k_ref**2) - (x - mu_ref) / (sigma_ref*k_ref)) * \
                                        erfc(-1*( ( (x-mu_ref)/sigma_ref -1/k_ref)/np.sqrt(2) ) )  
  
     integrand = min(px, pref)
     return(integrand)


def OI_uni_exg_numerical2(params_px, params_pref):
     # Note: We did the integration numerically, with a clever trick to deal with the infinite limits.

     k_x, mu_x, sigma_x = params_px
     k_ref, mu_ref, sigma_ref = params_pref

     ppf_0001 = [stats.exponnorm.ppf(0.0001, K = k_ref, loc = mu_ref, scale = sigma_ref), \
                    stats.exponnorm.ppf(0.0001, K = k_x, loc = mu_x, scale = sigma_x)]
     lower_lim = np.min(ppf_0001) # lower of the two ppf(0.001) points

     ppf_9999 = [stats.exponnorm.ppf(0.9999, K = k_ref, loc = mu_ref, scale = sigma_ref), \
                    stats.exponnorm.ppf(0.9999, K = k_x, loc = mu_x, scale = sigma_x)]
     upper_lim = np.max(ppf_9999) # higher of the two ppf(0.999) points

     oi_exg_numerical = scipy.integrate.quad(func_exg_OI_integrand2, lower_lim, upper_lim, args = (params_px, params_pref))[0]
     return(oi_exg_numerical)

def divide_array_into_n_parts(given_array, num_parts):
     len_array = len(given_array)
     num_elements_in_blocks = int(np.floor(len_array/num_parts))

     x = [given_array[i*num_elements_in_blocks:(i+1)*num_elements_in_blocks] for i in range(num_parts-1)] # every part except the last
     x.append(given_array[(num_parts-1)*num_elements_in_blocks:]) # the last part, which may have some more elements
     return(x)



# ------------------------------------------------------------------------------
# ----------------- Drift Diffusion model fitting ------------------------------
# ------------------------------------------------------------------------------

# ----------------------- Model Definition -------------------------------------

# Here we define our drift diffusion model and SPECIFY THE RANGES
# we use four parameters: drift, diffusion or noise, boundary separation, and non-decision time (i.e., time for perception and response execution).
# We will fit the model to each participant's data to get the parameter estimates for the participant.
# Note that the time unit used in the pyddm package is SECONDS.


def fit_ddm_model(data_to_fit, dt_to_set = 0.001):
     ddm_model_to_fit = pyddm.gddm(drift = "d", noise = 1.0, bound = "B", nondecision=0.2, starting_position="x0",
                         parameters={"d": (0, 20), "B": (0.1, 1.5), "x0": (-.8, .8)},
                         dt = dt_to_set)
     ddm_model_to_fit.fit(data_to_fit, lossfunction = pyddm.LossBIC, verbose = False)
     return(ddm_model_to_fit.parameters())

def fit_ddm_model_with_noise(data_to_fit, dt_to_set = 0.001):
     ddm_model_to_fit = pyddm.gddm(drift = "d", noise = "n", bound = "B", nondecision=0.2, starting_position="x0",
                         parameters={"d": (0, 20), "n": (0.1, 2.0), "B": (0.1, 1.5), "x0": (-.8, .8)},
                         dt = dt_to_set)
     ddm_model_to_fit.fit(data_to_fit, lossfunction = pyddm.LossBIC, verbose = False)
     return(ddm_model_to_fit.parameters())


# ----------------------- Preparing RT (and accuracy) data ---------------------

# Notably, the DDM needs RT data with accuracy labels.
# Also, the pyddm package needs the data in a certain format.
# We do these things here.

def generate_ddm_samples_from_real_data(rt_array, cor_array):
     rt_array = rt_array/1000
     cor_array = cor_array
     data_to_fit = pd.DataFrame(np.c_[rt_array, cor_array], columns = ["rt", "cor"])
     data_to_fit = pyddm.Sample.from_pandas_dataframe(data_to_fit, rt_column_name="rt", choice_column_name="cor")
     return(data_to_fit)



# --------------------------- Model fitting ------------------------------------


def fit_ddm_model_n_times(data_to_fit, n = 10, dt_to_set = 0.001, with_noise = 0):
     model_params = np.zeros([n, 5])

# An irritating point about the returned fit params from the pyddm package:
      # Different levels + the same things are labeled differently across levels.
      # I sidestepped it by creating two lists below.
     list1_of_keys = ["drift", "noise", "bound", "IC", "overlay"]
     list2_of_keys = ["drift", "noise", "B", "x0", "nondectime"]

     for i_run in range(n):
          if with_noise == 0: params_dict = fit_ddm_model(data_to_fit = data_to_fit, dt_to_set = dt_to_set)
          else: params_dict = fit_ddm_model_with_noise(data_to_fit = data_to_fit, dt_to_set = dt_to_set)
          model_params[i_run, :] = np.array([params_dict[k1][k2] + 0 for k1, k2 in zip(list1_of_keys, list2_of_keys)])
     # print(params_dict)

     return(model_params)
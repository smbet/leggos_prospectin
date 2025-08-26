import time, os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import expanduser
from astropy.table import Table, vstack
plt.rcParams.update({'xtick.major.pad': '7.0'})
plt.rcParams.update({'xtick.major.size': '7.5'})
plt.rcParams.update({'xtick.major.width': '1.5'})
plt.rcParams.update({'xtick.minor.pad': '7.0'})
plt.rcParams.update({'xtick.minor.size': '3.5'})
plt.rcParams.update({'xtick.minor.width': '1.0'})
plt.rcParams.update({'ytick.major.pad': '7.0'})
plt.rcParams.update({'ytick.major.size': '7.5'})
plt.rcParams.update({'ytick.major.width': '1.5'}) 
plt.rcParams.update({'ytick.minor.pad': '7.0'})
plt.rcParams.update({'ytick.minor.size': '3.5'})
plt.rcParams.update({'ytick.minor.width': '1.0'})
plt.rcParams.update({'xtick.color': 'k'})
plt.rcParams.update({'ytick.color': 'k'})
plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'axes.linewidth':2})
plt.rcParams.update({'patch.linewidth':2})
plt.rcParams['figure.figsize'] = (12, 8)

import re

import prospect
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from prospect.likelihood import lnlike_spec, lnlike_phot
from prospect.likelihood import chi_spec, chi_phot
from prospect.io import write_results as writer
from prospect.utils.obsutils import fix_obs
from prospect.models import SedModel
from prospect.fitting import fit_model
from prospect.plotting.utils import sample_posterior
from prospect.sources import CSPSpecBasis, FastStepBasis
from prospect.likelihood import lnlike_spec, lnlike_phot
from prospect.likelihood import chi_spec, chi_phot
from prospect.io import write_results as writer
from prospect.fitting import fit_model
from prospect.models.sedmodel import SedModel
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import PolySpecModel
from prospect.models import priors
import prospect.io.read_results as reader
#from prospect.plotting import corner as corn
from prospect.utils.obsutils import fix_obs
from prospect.io import write_results as writer
#from prospect.plotting.corner import quantile
from prospect.models import transforms

from multiprocessing import Pool
from contextlib import closing
import pickle

import pandas as pd
import sedpy
from astropy.io import ascii
from astropy.io import fits
import gc
import decimal

import cmasher as cmr

import pyregion
from matplotlib.colors import LogNorm
from adjustText import adjust_text

from sedpy import observate

from scipy import interpolate

from astropy.convolution import Gaussian1DKernel, convolve
from prospect.models import priors, SedModel
from scipy.stats import truncnorm
from scipy.integrate import cumtrapz # i can't believe they're deprecating cumtrapz :(

from astropy.cosmology import WMAP9 as cosmo
import random
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import argparse

#from lp_methods import*

### importing arguments first:
parser = argparse.ArgumentParser()
parser.add_argument("redshift", type=float, help="object redshift")
parser.add_argument("input_mag_path", type=str, help="input magnitude fits filepath")
parser.add_argument("input_h5_path", type=str, help="input h5 directory filepath")
parser.add_argument("input_image_path", type=str, help="input image fits filepath")
parser.add_argument("reg_path", type=str, help="input .reg filepath")
parser.add_argument("out_path", type=str, help="plot output directory filepath")
parser.add_argument("--starting_index", type=int, default=0, help="clump index you want to start at")
#parser.add_argument("--filterlist", type=list, 
#    default=['jwst_f090w','jwst_f162m','jwst_f210m','jwst_f277w','jwst_f300m','jwst_f444w'],
#    help="JWST filters that you have magnitudes of")
parser.add_argument("--mag_units", type=str, 
    help="magnitude and error units, currently not supported sry they need to be microjansky")

args = parser.parse_args()

zred = np.float64(args.redshift)
in_fits = args.input_mag_path
h5filedir = args.input_h5_path
in_img = args.input_image_path
reg_in = args.reg_path
outdir = args.out_path
ii_0 = args.starting_index

######## hey I was going to put all these methods in a separate script but that turned out to be
# taking a lot more time to do than I thought so this is good enough for now

serial = True

poly_calibration=1
add_neb=True
snr = 5

verbose = False

model_params = TemplateLibrary["continuity_sfh"]

# defining some methods

# not currently used but maybe useful so it stays for now
def t50_age(zred_all,t50_all):
    age_temp = []
    for ee, element2 in enumerate(t50_all):
        t50_temp = random.choices(t50_all, k=1)[0]
        #print(t50_temp)
        pz_temp = random.choices(zred_all, k=1)[0]
        tuniv_temp = cosmo.age(pz_temp).value
        age_temp.append(tuniv_temp - t50_temp)
    return age_temp

# dust fraction times dust2
def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

# traps... (any cis people reading: don't get the wrong idea)
def trap(x, y):
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2.

# next two are converting your input magnitudes into different units
def cgs_to_j3631(w2,f2):
    f2_new = 3.33564095E+04 * f2 * (w2**2)
    f2_new /= 3631
    return f2_new

def j3631_to_cgs(w2,f2):
    f2_new = f2*3631
    f2_new2 = f2_new*(2.99792458e-5)/(w2**2)
    return f2_new2

# method to make the observations object that prospector needs:
def load_obs(clump_ind, mags=None, mags_unc=None, snr=snr, redshift=20.0, ldist=10.0, **extras):

    # getting the filters we have photometry of:
    jwst = ['jwst_f090w','jwst_f162m','jwst_f210m','jwst_f277w','jwst_f300m','jwst_f444w']
    hst = []
    filternames = hst+jwst
    obs = {}
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # photometry and uncertainty
    obs["phot_wave"] = [f.wave_effective for f in obs["filters"]]
    #obs["maggies"] = cgs_to_j3631(np.array(obs["phot_wave"]),mags)
    #obs["maggies_unc"] = cgs_to_j3631(np.array(obs["phot_wave"]),mags_unc)
    ii = clump_ind # clump no.
    obs["maggies"] = np.array([[clumpdf.loc[ii].F090W_med*10**(-6)/3631],
                              [clumpdf.loc[ii].F162M_med*10**(-6)/3631],
                              [clumpdf.loc[ii].F210M_med*10**(-6)/3631],
                              [clumpdf.loc[ii].F277W_med*10**(-6)/3631],
                              [clumpdf.loc[ii].F300M_med*10**(-6)/3631],
                              [clumpdf.loc[ii].F444W_med*10**(-6)/3631],
                              ]).T[0]

    # And now we store the uncertainties (again in units of maggies)

    obs["maggies_unc"] = np.array([[clumpdf.loc[ii].F090err*10**(-6)/3631],
                              [clumpdf.loc[ii].F162err*10**(-6)/3631],
                              [clumpdf.loc[ii].F210err*10**(-6)/3631],
                              [clumpdf.loc[ii].F277err*10**(-6)/3631],
                              [clumpdf.loc[ii].F300err*10**(-6)/3631],
                              [clumpdf.loc[ii].F444err*10**(-6)/3631],
                              ]).T[0]
    obs["phot_mask"] = np.full_like(mags,True,dtype=bool)
    #obs["phot_mask"][np.isnan(mags)] = False
    #obs["phot_mask"][mags<0] = False
    
    snr_all = obs["maggies"]/obs["maggies_unc"]

    # sets a 5% uncertainty if the snr is greater than 50
    for t, element in enumerate(obs["filters"]):
        #if (element.wave_effective/(1+redshift)) < 1000:
        #    obs["phot_mask"][t] = False
        if snr_all[t] > 50:
            obs["maggies_unc"][t] = 0.05*obs["maggies"][t] 

    # spectra shit which we can ignore for now:
    obs["wavelength"] = None  # this would be a vector of wavelengths in angstroms if we had 
    obs["spectrum"] = None
    obs['unc'] = None  #spectral uncertainties are given here
    obs['mask'] = None

    return obs

# method for creating the age bins for our SFH:
def zred_to_agebins(zred=None,agebins=None,nbins_sfh=8,**extras):
    tuniv = cosmo.age(zred).value[0]*1e9
    tbinmax = (tuniv*0.9)
    agelims = [0.0,7.4772] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

# does what the method name says:
def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
    agebins = zred_to_agebins(zred=zred)
    logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()
    return m1 * coeffs

# see docstrings
def logsfr_ratios_to_masses_psb2(logmass=None, logsfr_ratios=None,
                                 logsfr_ratio_young=None, logsfr_ratio_old=None,
                                 tlast=None, tflex=None, nflex=None, nfixed=None,
                                 agebins=None,zred=None,**extras):
    """This is a modified version of logsfr_ratios_to_masses_flex above. This now
    assumes that there are nfixed fixed-edge timebins at the beginning of
    the universe, followed by nflex flexible timebins that each form an equal
    stellar mass. The final bin has variable width and variable SFR; the width
    of the bin is set by the parameter tlast.
    The major difference between this and the transform above is that
    logsfr_ratio_old is a vector.
    """

    # clip for numerical stability
    nflex = nflex[0]; nfixed = nfixed[0]
    logsfr_ratio_young = np.clip(logsfr_ratio_young[0], -7, 7)
    logsfr_ratio_old = np.clip(logsfr_ratio_old, -7, 7)
    syoung, sold = 10**logsfr_ratio_young, 10**logsfr_ratio_old
    sratios = 10.**np.clip(logsfr_ratios, -7, 7) # numerical issues...
    
    # get agebins
    abins = psb_logsfr_ratios_to_agebins2(logsfr_ratios=logsfr_ratios,
            agebins=agebins, tlast=tlast, tflex=tflex, nflex=nflex, nfixed=nfixed,zred=zred,**extras)

    # get find mass in each bin
    dtyoung, dt1 = (10**abins[:2, 1] - 10**abins[:2, 0])
    dtold = 10**abins[-nfixed-1:, 1] - 10**abins[-nfixed-1:, 0]
    old_factor = np.zeros(nfixed)
    for i in range(nfixed):
        old_factor[i] = (1. / np.prod(sold[:i+1]) * np.prod(dtold[1:i+2]) / np.prod(dtold[:i+1]))
    mbin = 10**logmass / (syoung*dtyoung/dt1 + np.sum(old_factor) + nflex)
    myoung = syoung * mbin * dtyoung / dt1
    mold = mbin * old_factor
    n_masses = np.full(nflex, mbin)

    return np.array(myoung.tolist() + n_masses.tolist() + mold.tolist())

# takes the age of the universe at the object's redshift and makes it 0.6 of that
def zred_to_tflex2(zred=None,tflex=None,**extras):
    #print("tflex container zred",zred)
    tuniv = cosmo.age(zred).value[0]    
    tflex=0.6*tuniv   #Make sure this is same as in the function below
    return tflex

def psb_logsfr_ratios_to_agebins2(logsfr_ratios=None, agebins=None,
                                 tlast=None, tflex=None, nflex=None, nfixed=None,zred=None,**extras):
    """This is a modified version of logsfr_ratios_to_agebins above. This now
    assumes that there are nfixed fixed-edge timebins at the beginning of
    the universe, followed by nflex flexible timebins that each form an equal
    stellar mass. The final bin has variable width and variable SFR; the width
    of the bin is set by the parameter tlast.
    For the flexible bins, we again use the equation:
        delta(t1) = tuniv  / (1 + SUM(n=1 to n=nbins-1) PROD(j=1 to j=n) Sn)
        where Sn = SFR(n) / SFR(n+1) and delta(t1) is width of youngest bin
    """

    # De-arrayfying values...
    tlast = tlast[0]; tflex = tflex[0]
    try: nflex = nflex[0]
    except IndexError: pass
    try: nfixed = nfixed[0]
    except IndexError: pass

    # numerical stability
    logsfr_ratios = np.clip(logsfr_ratios, -7, 7)
    
    ##################
    #Use redshift to recreate an agebin grid, such that the code below can utilise the fixed agebins definition
    #that also respects cosmology
    tuniv = cosmo.age(zred).value[0] 
    tflex = zred_to_tflex2(zred=zred,tflex=tflex,**extras)

    agelims = [1, tlast*1e9] + np.linspace((tlast + 0.1)*1e9, (tflex)*1e9, nflex).tolist() + \
    np.linspace(tflex*1e9, (tuniv)*1e9, nfixed+1)[1:].tolist()
    agebins_temp = np.array([agelims[:-1], agelims[1:]]).T
    agebins = np.log10(agebins_temp)
    ##################

    # flexible time is t_flex - youngest bin (= tlast, which we fit for)
    # this is also equal to tuniv - upper_time - lower_time
    tf = (tflex - tlast) * 1e9

    # figure out other bin sizes
    n_ratio = logsfr_ratios.shape[0]
    sfr_ratios = 10**logsfr_ratios
    dt1 = tf / (1 + np.sum([np.prod(sfr_ratios[:(i+1)]) for i in range(n_ratio)]))

    # translate into agelims vector (time bin edges)
    agelims = [1, (tlast*1e9), dt1+(tlast*1e9)]
    for i in range(n_ratio):
        agelims += [dt1*np.prod(sfr_ratios[:(i+1)]) + agelims[-1]]
    agelims += list(10**agebins[-nfixed:,1])  #Add the right end of the nfixed agebins to the agelims
    abins = np.log10([agelims[:-1], agelims[1:]]).T

    return abins

################################################################################

# here's the method where we make our initial model so this one has our initial parameter
# values and priors

# defining our sfh model:
model_params = TemplateLibrary["continuity_sfh"]

def build_model(model_params = model_params, add_neb=True):

    
    #######Initial Age Bin Definitions########
    tuniv_yr = cosmo.age(zred).value*1e9
    nbins_sfh = 8
    tbinmax = (tuniv_yr*0.8)
    agelims = [0.0,7.4772] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv_yr)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    agebins = agebins.T
    #print(10**agebins/1e9)
    ##########################################

    ### initial parameters and whether they're free or fixed and also priors
    
    #model_params['tau'] = {'N':1, 'isfree':True, 'init': 2.5,'prior':priors.LogUniform(mini=0.1, maxi=5.)}
    model_params['zred'] = {'N':1, 'isfree':False,'init':zred, 'prior': priors.TopHat(mini=zred-0.0005, maxi=zred+0.0005)}
    model_params["peraa"] = {"N": 1,"isfree": False,"init": False}
    #=============================================================================
    model_params["add_neb_emission"] ={"N": 1,"isfree": False,"init": True}
    model_params["add_neb_continuum"] ={"N": 1,"isfree": False,"init": True}
    model_params["nebemlineinspec"] = {"N": 1,"isfree": False,"init": True}
    #=============================================================================
    model_params['imf_type']['init'] = 1 # set IMF to chabrier (default is kroupa)
    model_params["logzsol"] = {"N": 1, "isfree": True, "init": -1.8, 'units': '$\\log (Z/Z_\\odot)$',
                                        'prior': priors.TopHat(mini=-2.0, maxi=0.20),}#'init_disp': 0.5, 'disp_floor': 0.1}
    #-----------------------------------------------------------------------------
    model_params['mass_units'] = {'name': 'mass_units', 'N': 1,'isfree': False,'init': 'mformed'}
    model_params['logmass'] = {'N':1, 'isfree':True,'init':8.0, 'prior': priors.TopHat(mini=4., maxi=10.)}#,'init_disp': 2.0, 'disp_floor': 0.5}
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['logsfr_ratios']['N'] = nbins_sfh-1
    model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                                                    scale=np.full(nbins_sfh-1,0.3),
                                                                    df=np.full(nbins_sfh-1,2))
    model_params['agebins']['depends_on'] = zred_to_agebins
    model_params['mass']['depends_on'] = logmass_to_masses
    #-----------------------------------------------------------------------------
    model_params['add_dust_emission'] = {'N':1, 'isfree':False, 'init':True}
    model_params['dust_type'] = {'N':1, 'isfree':False, 'init':4} #Kriek and Conroy?
    model_params['dust_index'] = {'N':1, 'isfree':False, 'init':-0.83,'prior':priors.TopHat(mini=-1, maxi=0.4)}
    #model_dict['dust2']['init'] = 0.5 / (2.5*np.log10(np.e))
    model_params['dust2']['init'] = 0.08 / (2.5*np.log10(np.e))
    #model_params['dust2']['prior'] = priors.TopHat(mini=0.0,maxi=2.5 / (2.5*np.log10(np.e)))
    model_params['dust2']['prior'] = priors.TopHat(mini=0.0,maxi=1.5)

    model_params['dust1'] = {'N':1, 'isfree':False, 'depends_on':to_dust1, 'init':1.0}
    model_params['dust1_fraction'] = {'N':1, 'isfree':False, 'init':1.0}
    #model_params['duste_gamma'] = {'N':1, 'isfree':False, 'init':0.01}
    #model_params['duste_umin'] = {'N':1, 'isfree':False, 'init':1.0}
    #model_params['duste_qpah'] = {'N':1, 'isfree':False, 'init':2.0}

    model_params["add_igm_absorption"] ={"N": 1,"isfree": False,"init": True}
    #model_params['igm_factor'] = {'N':1, 'isfree':True, 'init':1.0,'prior':priors.TopHat(mini=0.1,maxi=10)}

    #model_params["gas_logu"] = {"N": 1, 'isfree': True,'init': -2., 'units': r"Q_H/N_H",
    #                        'prior': priors.TopHat(mini=-4, maxi=-1)}
    model_params["gas_logz"] = {"N": 1, 'isfree': True,'init': -1., 'units': r"Q_H/N_H",
                            'prior': priors.TopHat(mini=-2, maxi=0.2)}

    model_params['fagn'] = {'N':1, 'isfree':True, 'init':0.3,'prior':priors.LogUniform(mini=0.01,maxi=0.5)}
    model_params['agn_tau'] = {'N':1, 'isfree':True, 'init':10.0,'prior':priors.TopHat(mini=6,maxi=20.0)}
    
    ################################################################################################
   
    return model_params 

# loads object to create our model's stellar populations
def load_sps(zcontinuous=1, **extras):
    from prospect.sources import CSPSpecBasis, FastStepBasis
    #sps = CSPSpecBasis(zcontinuous=zcontinuous,reserved_params=["sigma_smooth"])
    sps = FastStepBasis(zcontinuous=zcontinuous,reserved_params=["sigma_smooth"])
    return sps

# couldn't find a built in thing to do this which is weird but whatever
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

########

#---------------------------------------------------------------------------------------------------
#
# grabbing the magnitudes file:

home = expanduser('~')
path = in_fits
filepath = glob(path, recursive=True)
clumptable = Table.read(filepath[0], format='fits')
clumpdf = clumptable.to_pandas()
# need to convert the 16 and 84th percentile values into errors real quick:
clumpdf['F090err'] = (clumpdf.F090W_84-clumpdf.F090W_16)/2
clumpdf['F162err'] = (clumpdf.F162M_84-clumpdf.F162M_16)/2
clumpdf['F210err'] = (clumpdf.F210M_84-clumpdf.F210M_16)/2
clumpdf['F277err'] = (clumpdf.F277W_84-clumpdf.F277W_16)/2
clumpdf['F300err'] = (clumpdf.F300M_84-clumpdf.F300M_16)/2
clumpdf['F444err'] = (clumpdf.F444W_84-clumpdf.F444W_16)/2

run_params = {}
run_params["verbose"] = verbose
#run_params["dimensions"] = len(model.theta)
run_params["optimize"] = False
run_params["emcee"] = False
run_params["dynesty"] = True
run_params['nested_bound'] = 'multi'
run_params['nested_sample'] = 'rwalk'
run_params['nested_maxbatch'] = None

run_params['nested_nlive_init'] = 600 # change this to 900 if speed is no object
run_params['nested_nlive_batch'] = 200
run_params['nested_target_n_effective'] = 15000
#run_params['nested_nlive_init'] = 400 #maybe keep this to 200 (this is slowing it down i think?)
run_params['nested_nlive_batch'] = 100
run_params['nested_target_n_effective'] = 100
run_params['nested_weight_kwargs'] = {'pfrac': 1.0}
run_params['nested_dlogz_init'] = 0.05

run_params['object_redshift'] = zred
list_of_spec_lists = []
sfh_and_theta = []


# # # # # # # # # 
#for loop where individual clump plots are made

for ii in range(len(clumpdf)-ii_0):
    
    run_params['clump_ind']=ii+ii_0

    # checking object has a redshift greater than 0
    if ii==0:
        if zred>0:
            tuniv = cosmo.age(zred).value
            print("Age of Universe at redshift is =", tuniv)
            print()
        else:
            raise(ValueError('Quitting fit because zred value not greater than 0.'))

    obs = load_obs(**run_params)
    sps = load_sps(**run_params)
    model_params = build_model()

    model = PolySpecModel(model_params)
    theta = model.theta.copy()
    
    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    # photometric effective wavelengths
    wphot = obs["phot_wave"]
    # spectroscopic wavelengths
    if obs["wavelength"] is None:
        # *restframe* spectral wavelengths, since obs["wavelength"] is None
        wspec = sps.wavelengths
        wspec *= a #redshift them
    else:
        wspec = obs["wavelength"]

    try:
        hfile = h5filedir+'1527clump_'+str(run_params['clump_ind'])+'**_dynesty.h5'
        hfilepath = glob(hfile, recursive=True)[0] # need to edit this so it just returns the most recent

        result, obs, _ = reader.results_from(hfilepath, dangerous=False)

        nparam = len(model.theta)
        imax = np.argmax(result['lnprobability'])
        #Dynesty
        theta_max = result["chain"][imax, :]
    except:

        print('can\'t find the h5 file with the baysian analysis stuff for clump {0:}'.format(str(run_params['clump_ind'])))

    mspec_max1, mphot_max1, mextra_max1 = model.mean_model(theta_max, obs, sps=sps)

    speclist = []
    speclist.append(wspec)
    speclist.append(mspec_max1)

    model_wave_max = wspec
    obs_nospec = obs.copy()
    obs_nospec['wavelength'] = model_wave_max
    obs_nospec['spectrum'] = None

    nsample1 = 1000
    theta_arr = sample_posterior(result["chain"], weights=result.get("weights", None), nsample=nsample1)

    agebins=[]
    agebins = model.params['agebins']
    logr = np.array(theta_max[model.theta_index['logsfr_ratios']])
    logmass = np.array(theta_max[model.theta_index['logmass']])
    agebins = prospect.models.transforms.logsfr_ratios_to_agebins(logsfr_ratios=logr,
                    logmass=logmass,agebins=agebins)
    masses = prospect.models.transforms.logsfr_ratios_to_masses(logsfr_ratios=logr,
                    logmass=logmass, agebins=agebins)
    dt = 10**agebins[:, 1] - 10**agebins[:, 0]
    agemid_max = 0.5*(10**agebins[:, 1] + 10**agebins[:, 0])
    sfrs_max = masses/dt
    agebin_interp_max = np.arange(np.min(10**agebins/1e9), np.max(10**agebins/1e9), 0.001)
    sfr_interp = np.zeros(len(agebin_interp_max))

    zred_minimum = zred

    tuniv_maximum = cosmo.age(zred_minimum).value    
    #agebin_interp = []
    #agebin_interp = np.arange(1e-9,tuniv_maximum,0.001)
    agebin_interp = np.arange(1e-9,tuniv_maximum,0.025)

    sfr_array_max = sfr_interp

    agebin_centers_max = agebin_interp_max[:-1] + np.diff(agebin_interp_max)[0]  #age bins
    cumulative_mass_formed_max = cumtrapz(sfr_interp[::-1], agebin_interp_max*1e9) #reversing the array from BB to epoch of obs
    #t50_max = agebin_centers_max[cumulative_mass_formed_max/cumulative_mass_formed_max[-1]>0.5][0]
    
    sfr_array = []
    spec_array_calib = []
    spec_array_nocalib = []
    phot_array = []
    
    SFR_all = []
    fburst_all = []
    logM_all = []
    logZ_all = []
    dust2_all = []
    sigma_all = []
    jitter_all = []
    sigma_gas_all = []
    t10_all = []
    t50_all = []
    t90_all = []
    zred_all = []
    model_wave_all = []
    agebin_interp_all = []
    agebin_centers_all = []
    agemid_all = []
    sfrs_mid_all = []
    t50_coarse = []
    
    
    SFR_all_alt=[]
    agebin_centers_all_alt=[]
    t50_all_alt = []
    
    #t_young = 0.5  #Gyr time for which you would like to calculate fburst
    
    index = 0
    #agebin_interp = []
    for theta in theta_arr:
        ####
        #zred = np.array(theta[model.theta_index['zred']])
        #zred_all.append(zred[0])
        zred_all.append(zred)
    
        tuniv = cosmo.age(zred).value    
        a = 1.0 + zred # cosmological redshifting
        wphot = obs["phot_wave"]
        if obs["wavelength"] is None:
            wspec = sps.wavelengths.copy()
            wspec *= a #redshift them
        else:
            wspec = obs["wavelength"]
        model_wave = wspec
        obs_nospec = obs.copy()
        obs_nospec['wavelength'] = model_wave
        obs_nospec['spectrum'] = None
        ####
    
        logr = np.array(theta[model.theta_index['logsfr_ratios']])
        logmass = np.array(theta[model.theta_index['logmass']])
    
        agebins = prospect.models.transforms.logsfr_ratios_to_agebins(logsfr_ratios=logr,
            logmass=logmass,agebins=agebins)
    
        masses = prospect.models.transforms.logsfr_ratios_to_masses(logsfr_ratios=logr,
            logmass=logmass, agebins=agebins)
    
    
        dt = 10**agebins[:, 1] - 10**agebins[:, 0]
        agemid = 0.5*(10**agebins[:, 1] + 10**agebins[:, 0])
        agemid = agebins[:-1] + np.diff(agebins)[0]
        agemid_all.append(agemid)
        sfrs = masses/dt
        sfrs_mid_all.append(sfrs)
    
        sfr_interp = np.zeros(len(agebin_interp))
        for age,sfr in zip(agebins, sfrs):
            sfr_interp[(agebin_interp>=(10**age[0]/1e9)) & (agebin_interp<(10**age[1]/1e9))] = sfr
    
        try:
            #print("ID Number, loop number:",id_number,index)
            model_spec_calib, model_phot_calib, mass_loss =  model.predict(theta, obs, sps)
            model_spec_nocalib, model_phot = model_spec_calib, model_phot_calib
    
        except ValueError:
            model_spec_calib = np.nan * np.zeros(len(wave_obs))
            model_spec_nocalib = np.nan*np.zeros(len(model_wave))
            model_phot = np.nan * np.zeros(len(phot_wl))
            sfr_interp = np.nan * np.zeros(len(agebin_interp))
    
        if np.sum(np.isfinite(model_spec_calib))==0:
            model_spec_calib = np.nan * np.zeros(len(wave_obs))
            model_spec_nocalib = np.nan * np.zeros(len(model_wave))
            model_phot = np.nan * np.zeros(len(phot_wl))
            sfr_interp = np.nan * np.zeros(len(agebin_interp))

        sfr_array.append(sfr_interp)
        spec_array_calib.append(model_spec_calib)
        spec_array_nocalib.append(model_spec_nocalib)
        phot_array.append(model_phot)
    
        #----------------------------------
        #model_spec_calib, model_phot_calib, mass_loss =  model.predict(theta, obs, sps)
    
        log_M_temp = logmass + np.log10(mass_loss)
        logM_all.append( log_M_temp[0] )
        logZ_all.append(theta[model.theta_index['logzsol']][0])
        dust2_all.append(theta[model.theta_index['dust2']][0])
    
        mtot = np.trapz(sfr_interp, agebin_interp*1e9)
        #myoung = np.trapz(sfr_interp[agebin_interp<t_young], agebin_interp[agebin_interp<t_young]*1e9)
    
        SFR_all.append(np.log10(sfr_interp[0]))
        agebin_centers = agebin_interp[:-1] + np.diff(agebin_interp)[0]  #age bins
        agebin_centers_all.append(agebin_centers)
        cumulative_mass_formed = cumtrapz(sfr_interp[::-1], agebin_interp*1e9) #reversing the array from BB to epoch of obs
    
        #####for alternate interpolation#######
        agebin_interp_intermediate = np.arange(np.min(10**agebins/1e9), np.max(10**agebins/1e9), 0.025)
        sfr_interp_intermediate = np.zeros(len(agebin_interp_intermediate))

        for age,sfr in zip(agebins, sfrs):
            sfr_interp_intermediate[(agebin_interp_intermediate>=(10**age[0]/1e9)) & (agebin_interp_intermediate<(10**age[1]/1e9))] = sfr

        SFR_all_alt.append(np.log10(sfr_interp_intermediate[0]))
        agebin_centers =agebin_interp_intermediate[:-1] + np.diff(agebin_interp_intermediate)[0]  #age bins
        agebin_centers_all_alt.append(agebin_centers)
        cumulative_mass_formed = cumtrapz(sfr_interp_intermediate[::-1], agebin_interp_intermediate*1e9) #reversing the array from BB to epoch of obs
        
    
        model_wave_all.append(model_wave)
        agebin_interp_all.append(agebin_interp)
    
        index+=1

    phot_array_max = model_phot
    spec_array_calib_max = model_spec_calib
    spec_array_nocalib_max = model_spec_nocalib
    
    model_wave_all = np.array(model_wave_all)
    sfr_array = np.array(sfr_array)
    spec_array_calib = np.array(spec_array_calib)
    spec_array_nocalib = np.array(spec_array_nocalib)
    phot_array = np.array(phot_array)
    zred_all = np.array(zred_all)
    
    agebin_interp_all = np.array(agebin_interp_all)
    agebin_centers_all = np.array(agebin_centers_all)
    
    SFR_all_alt=np.array(SFR_all_alt)
    agebin_centers_all_alt=np.array(agebin_centers_all_alt)
    
    sfrs_mid_all = np.array(sfrs_mid_all)
    agemid_all = np.array(agemid_all)
    
    logM_16, logM_50, logM_84 = np.nanpercentile(logM_all, [16,50,84])
    sfr_16, sfr_50, sfr_84 = np.nanpercentile(SFR_all, [16,50,84])
    
    sSFR_all2 = np.array(SFR_all) - np.array(logM_all)
    sSFR_all = sSFR_all2[~np.isnan(sSFR_all2)]

    f2_cgs_max = j3631_to_cgs(model_wave_max,spec_array_nocalib_max)
    f2_phot_cgs_max = j3631_to_cgs(np.array(wphot), np.array(obs['maggies']))
    f2_model_cgs_max = j3631_to_cgs(np.array(wphot), phot_array_max)
    
    f2_cgs = j3631_to_cgs(model_wave_all,spec_array_nocalib)
    f2_phot_cgs = j3631_to_cgs(np.array(wphot), np.array(obs['maggies']))
    f2_model_cgs = j3631_to_cgs(np.array(wphot), phot_array)

    # getting time at which median sfr happens
    sfharray = np.nanmedian(sfr_array, axis=0)
    mediansfr = find_nearest(sfharray,np.nanmedian(sfharray))
    median_sfr_time = np.median(agebin_interp[np.where(sfharray==mediansfr)])

    sfr_90=find_nearest(np.nanmedian(sfr_array, axis=0),0.9*np.nanmax(np.nanmedian(sfr_array, axis=0)))
    sfr_time_90=np.median(agebin_interp[np.where(np.nanmedian(sfr_array, axis=0)==sfr_90)])
    sfr_10=find_nearest(np.nanmedian(sfr_array, axis=0),0.1*np.nanmax(np.nanmedian(sfr_array, axis=0)))
    sfr_time_10=np.median(agebin_interp[np.where(np.nanmedian(sfr_array, axis=0)==sfr_10)])

    # saving params to a list:

    theta_keys = ['logzsol','dust2','logmass','logsfratio1','logsfratio2','logsfratio3','logsfratio4','gaslogz','fagn','agn_tau']

    param_text = []
    
    sfr_array_list=[run_params['clump_ind'],median_sfr_time]
    
    for ii in range(len(theta_keys)):
        param_text.append(theta_keys[ii]+' = ' + str(theta[ii])[0:5])
        sfr_array_list.append(theta[ii])

    sfh_and_theta.append(sfr_array_list)

    # making and then saving a SFH plot:

    fig,ax3 = plt.subplots(figsize=(20,10)) # subplot numbering conventions don't make sense sry they come out okay though
    plt.title('1527 Clump {0:} SFH'.format(str(run_params['clump_ind'])))
    left, bottom, width, height = [0.45, 0.7, 0.15, 0.15]
    ax4 = fig.add_axes([left, bottom, width, height])
    ax5 = fig.add_axes([0.65, bottom, width, height])
    
    ax4.tick_params(axis='both', which='major', labelsize=20)
    ax5.tick_params(axis='both', which='major', labelsize=20)
    #--------------------------
    
    ax4.hist(logM_all,bins=10,histtype='stepfilled',color='xkcd:barbie pink')
    ax4.set_xlabel(r'log(M$_{\star} /$ M$_\odot)$', fontsize=18)
    ax4.set_xlim(np.percentile(logM_all,0.01),np.percentile(logM_all,99.99))
    ax5.hist(sSFR_all,bins=10,histtype='stepfilled',color='xkcd:barbie pink')
    ax5.set_xlabel('log(sSFR) (Gyr$^{-1}$)', fontsize=18)
    ax5.set_xlim(np.percentile(sSFR_all,0.1),np.percentile(sSFR_all,99.9))
    #--------------------------
    
    ax3.plot(agebin_interp, np.nanmedian(sfr_array, axis=0), color='turquoise',lw=5)
    ax3.fill_between(agebin_interp, np.nanpercentile(sfr_array, 16,axis=0), np.nanpercentile(sfr_array, 84,axis=0),
                    color='grey', alpha=0.2)
    ax3.fill_between(agebin_interp, np.nanpercentile(sfr_array, 2.5,axis=0), np.nanpercentile(sfr_array, 97.5,axis=0),
                    color='grey', alpha=0.1)
    ax3.set_xlabel('Lookback Time (Gyr)', fontsize=18)
    ax3.set_ylabel(r'$\mu \times$ SFR (M$_\odot$/yr)', fontsize=18)
    ax3.set_xscale('log')
    ax3.set_xlim(5e-3,5.0)
    #ax3.set_ylim(-0.5, np.nanpercentile(np.nanpercentile(sfr_array, 84,axis=0),97.5))
    #ax3.plot(np.append(agebin_interp_max,agebin_interp_max[-1]+0.001), np.append(sfr_array_max,0),\
    #color='xkcd:barbie pink',lw=1,ls='--')
    
    pfile = outdir+'1527clump_'+str(run_params['clump_ind'])+'SFHplot.png'
    
    plt.savefig(pfile,dpi=300)
    plt.close()

    # making and saving best fit model plot:

    obs_markersize = 15

    fig,ax6 = plt.subplots(figsize=(20,10))

    for t, element in enumerate(obs['maggies']):
    
        if obs["phot_mask"][t] == True:
            ax6.errorbar(wphot[t], f2_phot_cgs[t],
                        yerr=j3631_to_cgs(wphot[t],obs['maggies_unc'][t]),
                        zorder=4,marker='o', markersize=obs_markersize, alpha=0.8, ls='', lw=3,
                    ecolor='black', markerfacecolor='black', markeredgecolor='black', markeredgewidth=3)
        else:
            ax6.errorbar(wphot[t], f2_phot_cgs[t],
                        yerr=j3631_to_cgs(wphot[t],obs['maggies_unc'][t]),
                        zorder=4,marker='o', markersize=obs_markersize, alpha=0.8, ls='', lw=3,
                    ecolor='black', markerfacecolor='grey', markeredgecolor='black', markeredgewidth=3)

    ax6.errorbar(wphot[-1], f2_phot_cgs[-1], yerr=j3631_to_cgs(wphot[-1],obs['maggies_unc'][-1]),
                    marker='o', markersize=obs_markersize, alpha=0.8, ls='', lw=3,
                    ecolor='black', markerfacecolor='black', markeredgecolor='black', markeredgewidth=3,label='Observed Photometry')
    
    cond_mask = (obs["phot_mask"]==True)
    
    ax6.errorbar(np.array(wphot)[cond_mask], np.nanmedian(f2_model_cgs, axis=0)[cond_mask],
            [np.nanmedian(f2_model_cgs, axis=0)[cond_mask] - np.nanpercentile(f2_model_cgs, 16, axis=0)[cond_mask],
                np.nanpercentile(f2_model_cgs, 84, axis=0)[cond_mask] - np.nanmedian(f2_model_cgs, axis=0)[cond_mask]], 
                marker='s',markersize=10, alpha=1.0, ls='', lw=5,
            markerfacecolor='orange', markeredgecolor='black', ecolor='orange',
            markeredgewidth=3,zorder=5,label='Best fit Model Photometry')
    
    for k,element in enumerate(model_wave_all[::2]):
        ax6.plot(model_wave_all[k],f2_cgs[k],color='rosybrown',alpha=0.1)
    ax6.plot(model_wave_max,f2_cgs_max,lw=2,zorder=4,color='xkcd:barbie pink',label='Best fit SED')
    
    ax6.set_xlim(3000, 60000)
    ax6.set_ylim([np.nanmin(f2_phot_cgs[obs["phot_mask"] == True])*0.5, 
            np.nanmedian(f2_cgs_max)*5])
    
    ax6.set_xlabel(r'Wavelength ($\AA$)', fontsize=18)
    ax6.set_ylabel(r'F$_\lambda$ (erg cm$^{-2}$ s$^{-1}$ A$^{-1}$ )', fontsize=18)
    #ax6.set_xscale('log') #leaving it linear for now
    ax6.legend(fontsize=15)
    
    plt.title('Best fit model for 1527 clump {0:}'.format(str(run_params['clump_ind'])))

    bfile = outdir+'1527clump_'+str(run_params['clump_ind'])+'_best_fit_plot.png'

    plt.savefig(bfile,dpi=300)
    plt.close()

    # making and then saving a SFH + best fit plot:

    fig,(ax3,ax6) = plt.subplots(1,2,figsize=(20,8))
    
    fig.suptitle('1527 Clump {0:}'.format(str(run_params['clump_ind'])))
    
    ax3.plot(agebin_interp, np.nanmedian(sfr_array, axis=0), color='turquoise',lw=5)
    ax3.fill_between(agebin_interp, np.nanpercentile(sfr_array, 16,axis=0), np.nanpercentile(sfr_array, 84,axis=0),
                    color='grey', alpha=0.2)
    ax3.fill_between(agebin_interp, np.nanpercentile(sfr_array, 2.5,axis=0), np.nanpercentile(sfr_array, 97.5,axis=0),
                    color='grey', alpha=0.1)
    ax3.set_xlabel('Lookback Time (Gyr)', fontsize=18)
    ax3.set_ylabel(r'$\mu \times$ SFR (M$_\odot$/yr)', fontsize=18)
    ax3.set_xscale('log')
    ax3.set_xlim(5e-3,5.0)
    
    obs_markersize = 15
    
    for t, element in enumerate(obs['maggies']):
    
        if obs["phot_mask"][t] == True:
            ax6.errorbar(wphot[t], f2_phot_cgs[t],
                        yerr=j3631_to_cgs(wphot[t],obs['maggies_unc'][t]),
                        zorder=4,marker='o', markersize=obs_markersize, alpha=0.8, ls='', lw=3,
                    ecolor='black', markerfacecolor='black', markeredgecolor='black', markeredgewidth=3)
        else:
            ax6.errorbar(wphot[t], f2_phot_cgs[t],
                        yerr=j3631_to_cgs(wphot[t],obs['maggies_unc'][t]),
                        zorder=4,marker='o', markersize=obs_markersize, alpha=0.8, ls='', lw=3,
                    ecolor='black', markerfacecolor='grey', markeredgecolor='black', markeredgewidth=3)
    
    ax6.errorbar(wphot[-1], f2_phot_cgs[-1], yerr=j3631_to_cgs(wphot[-1],obs['maggies_unc'][-1]),
                    marker='o', markersize=obs_markersize, alpha=0.8, ls='', lw=3,
                    ecolor='black', markerfacecolor='black', markeredgecolor='black', markeredgewidth=3,label='Observed Photometry')
    
    cond_mask = (obs["phot_mask"]==True)
    
    ax6.errorbar(np.array(wphot)[cond_mask], np.nanmedian(f2_model_cgs, axis=0)[cond_mask],
            [np.nanmedian(f2_model_cgs, axis=0)[cond_mask] - np.nanpercentile(f2_model_cgs, 16, axis=0)[cond_mask],
                np.nanpercentile(f2_model_cgs, 84, axis=0)[cond_mask] - np.nanmedian(f2_model_cgs, axis=0)[cond_mask]], 
                marker='s',markersize=10, alpha=1.0, ls='', lw=5,
            markerfacecolor='orange', markeredgecolor='black', ecolor='orange',
            markeredgewidth=3,zorder=5,label='Best fit Model Photometry')
    
    for k,element in enumerate(model_wave_all[::2]):
        ax6.plot(model_wave_all[k],f2_cgs[k],color='rosybrown',alpha=0.1)
    ax6.plot(model_wave_max,f2_cgs_max,lw=2,zorder=4,color='xkcd:barbie pink',label='Best fit SED')
    
    ax6.set_xlim(3000, 60000)
    ax6.set_ylim([np.nanmin(f2_phot_cgs[obs["phot_mask"] == True])*0.5, 
            np.nanmedian(f2_cgs_max)*5])
    
    ax6.set_xlabel(r'Wavelength ($\AA$)', fontsize=18)
    ax6.set_ylabel(r'F$_\lambda$ (erg cm$^{-2}$ s$^{-1}$ A$^{-1}$ )', fontsize=18)
    ax6.legend(fontsize=15)
    
    cfile = outdir+'1527clump_'+str(run_params['clump_ind'])+'.png'
        
    plt.savefig(cfile,dpi=300)
    plt.close()

    # sfh with model inset:

    fig,ax3 = plt.subplots(figsize=(20,10))
    plt.title('1527 Clump {0:}'.format(str(run_params['clump_ind'])))
    left, bottom, width, height = [0.47, 0.55, 0.25, 0.25]
    ax6 = fig.add_axes([left, bottom, width, height])
    
    for k,element in enumerate(model_wave_all[::2]):
        ax6.plot(model_wave_all[k],f2_cgs[k],color='rosybrown',alpha=0.1)
    ax6.errorbar(wphot[-1], f2_phot_cgs[-1], yerr=j3631_to_cgs(wphot[-1],obs['maggies_unc'][-1]),
                    marker='o', markersize=obs_markersize, alpha=0.8, ls='', lw=3,
                    ecolor='black', markerfacecolor='black', markeredgecolor='black', markeredgewidth=3,label='Observed Photometry')
    
    cond_mask = (obs["phot_mask"]==True)
    
    ax6.errorbar(np.array(wphot)[cond_mask], np.nanmedian(f2_model_cgs, axis=0)[cond_mask],
            [np.nanmedian(f2_model_cgs, axis=0)[cond_mask] - np.nanpercentile(f2_model_cgs, 16, axis=0)[cond_mask],
                np.nanpercentile(f2_model_cgs, 84, axis=0)[cond_mask] - np.nanmedian(f2_model_cgs, axis=0)[cond_mask]], 
                marker='s',markersize=10, alpha=1.0, ls='', lw=5,
            markerfacecolor='orange', markeredgecolor='black', ecolor='orange',
            markeredgewidth=3,zorder=5,label='Best fit Model Photometry')
    
    for k,element in enumerate(model_wave_all[::2]):
        ax6.plot(model_wave_all[k],f2_cgs[k],color='rosybrown',alpha=0.1)
    ax6.plot(model_wave_max,f2_cgs_max,lw=2,zorder=4,color='xkcd:barbie pink',label='Best fit SED')
    
    ax6.set_xlim(3000, 60000)
    ax6.set_ylim([np.nanmin(f2_phot_cgs[obs["phot_mask"] == True])*0.5, 
            np.nanmedian(f2_cgs_max)*5])
    
    ax6.set_xlabel(r'Wavelength ($\AA$)', fontsize=22)
    ax6.set_ylabel(r'F$_\lambda$ (erg cm$^{-2}$ s$^{-1}$ A$^{-1}$ )', fontsize=22)
    ax6.legend(fontsize=8,loc='upper right')
    
    #--------------------------
    
    ax3.plot(agebin_interp, np.nanmedian(sfr_array, axis=0), color='turquoise',lw=5,label='Median SFH')
    ax3.fill_between(agebin_interp, np.nanpercentile(sfr_array, 16,axis=0), np.nanpercentile(sfr_array, 84,axis=0),
                    color='grey', alpha=0.2)
    ax3.fill_between(agebin_interp, np.nanpercentile(sfr_array, 2.5,axis=0), np.nanpercentile(sfr_array, 97.5,axis=0),
                    color='grey', alpha=0.1)
    peaksfr = np.median(agebin_interp[np.where(np.nanmedian(sfr_array, axis=0)==np.nanmax(np.nanmedian(sfr_array, axis=0)))])
    ax3.axvline(x=peaksfr,linewidth=16,c='xkcd:darkish blue',alpha=0.8,linestyle='--',label='Peak SFR')
    ax3.axvline(x=np.max(agebin_interp)-median_sfr_time,linewidth=16,c='xkcd:burgundy',alpha=0.8,linestyle='--',label=r'$t_{50}$')
    ax3.legend(fontsize=16,loc='center right')
    ax3.set_xlabel('Lookback Time (Gyr)', fontsize=18)
    ax3.set_ylabel(r'$\mu \times$ SFR (M$_\odot$/yr)', fontsize=18)
    ax3.set_xscale('log')
    ax3.set_xlim(5e-3,5.0)
    
    gpfile = outdir+'1527clump_'+str(run_params['clump_ind'])+'_combined_plots.png'
        
    plt.savefig(gpfile,dpi=300)
    plt.close()

# done with individual clump plots, onto collective ones:

try:
    # create dataframe with SFR_array_list and old dataframe with magnitudes if it isn't already there

    df_sfh = pd.read_csv(outdir+'1527bestphotmodel.csv')
    
except:
    df_sfh = pd.DataFrame(sfh_and_theta,columns=[['index','t_median_sfr']+theta_keys])
    df_sfh.to_csv(outdir+'1527bestphotmodel.csv', index=False)
    df_sfh = pd.read_csv(outdir+'1527bestphotmodel.csv')

# loading in fits image file for pretty pictures:

#fitspath0 = os.path.join(home,'python','prospectin','**','./S1527_F090W_noBCG.fits')
fitspath0 = in_img
fitspath1 = glob(fitspath0, recursive=True)
hdu = fits.open(fitspath1[0])
f090w_phot = hdu[0].data.copy()
head = hdu[0].header.copy()
hdu.close()

# loading in clump regions file:

regionspath = reg_in
regions = glob(regionspath, recursive=True)
reg_shape_list = pyregion.open(regions[0]).as_imagecoord(header=head)

reg_array = np.array([reg_shape_list[jj].coord_list for jj in range(len(reg_shape_list))])

# x and y positions of clump region centers
clumpx = reg_array.T[0]
clumpy = reg_array.T[1]

### clump regions plot:

fig, ax = plt.subplots(1,1)

fig.set_size_inches(8,6)

fig.tight_layout()

ax.imshow(f090w_phot, norm=LogNorm(vmin=0.05), origin='lower', cmap='cmr.guppy')

patch_list, artist_list = reg_shape_list.get_mpl_patches_texts()

for p in patch_list:
    ax.add_patch(p)
for t in artist_list:
    ax.add_artist(t)

ax.set_xlim(40,350)
ax.set_ylim(100,240)

texts = [ax.text(clumpx[i], clumpy[i], '%s' %i, ha='center', va='center', fontsize=12) for i in range(len(reg_array))]
# and just make x and y the ellipse centers
adjust_text(texts, expand=(1.2, 2), arrowprops=dict(arrowstyle="-", color='k', lw=0.8))

ax.set_title('SGAS 1527 Clump Regions',size=26)
ax.axis('off')

mapfile = outdir+'1527clump_map.png'
    
plt.savefig(mapfile,dpi=300)
plt.close()

### color mapping to some of the parameters:

# starting with mass

fig, ax = plt.subplots(1,1)

fig.tight_layout()

# plotting UV photometry as background

ax.imshow(f090w_phot, norm=LogNorm(vmin=0.), origin='lower', cmap='cmr.sunburst')

# putting in the regions

patch_list, artist_list = reg_shape_list.get_mpl_patches_texts()

for p in patch_list:
    ax.add_patch(p)
for t in artist_list:
    ax.add_artist(t)

# just making little dots with plt.scatter for the clumps because that's wayyyy simpler than plotting ellipses

colormask = df_sfh.logmass

plt.scatter(clumpx,clumpy,c=colormask,cmap='cmr.lavender',s=50,alpha=0.95)
cb=plt.colorbar(shrink=0.57)
cb.set_ticks(ticks=np.round(np.linspace(np.min(colormask),np.max(colormask),num=6),decimals=2))
cb.set_label(r'$\log{\frac{M}{M_{\odot}}}$',size=24)
ax.set_title('SGAS 1527 clump mass',size=26)
    
ax.axis('off')

# adding labels for the clump indices so we know which one is which

texts = [ax.text(clumpx[i], clumpy[i], '%s' %i, ha='center', va='center', fontsize=7) for i in range(len(reg_array))]
adjust_text(texts, expand=(1.2, 2), arrowprops=dict(arrowstyle="-", color='k', lw=0.8))

ax.set_xlim(40,350)
ax.set_ylim(100,240)

massmapfile = outdir+'1527clumpmass_map.png';

plt.savefig(massmapfile,dpi=300)

plt.close()

# metallicity plot:

fig, ax = plt.subplots(1,1)

fig.tight_layout()

ax.imshow(f090w_phot, norm=LogNorm(vmin=0.), origin='lower', cmap='cmr.sunburst')

# putting in the regions

patch_list, artist_list = reg_shape_list.get_mpl_patches_texts()

for p in patch_list:
    ax.add_patch(p)
for t in artist_list:
    ax.add_artist(t)

# just making little dots with plt.scatter for the clumps because that's wayyyy simpler than plotting ellipses

colormask = df_sfh.logzsol

plt.scatter(clumpx,clumpy,c=colormask,cmap='cmr.lavender',s=50,alpha=0.95)
cb=plt.colorbar(shrink=0.57)
cb.set_ticks(ticks=np.round(np.linspace(np.min(colormask),np.max(colormask),num=6),decimals=2))
cb.set_label(r'$\log{\frac{Z}{Z_{\odot}}}$',size=24)
ax.set_title('SGAS 1527 clump metallicity',size=26)
    
ax.axis('off')

# adding labels for the clump indices so we know which one is which

texts = [ax.text(clumpx[i], clumpy[i], '%s' %i, ha='center', va='center', fontsize=7) for i in range(len(reg_array))]
adjust_text(texts, expand=(1.2, 2), arrowprops=dict(arrowstyle="-", color='k', lw=0.8))

ax.set_xlim(40,350)
ax.set_ylim(100,240)

zmapfile = outdir+'1527metallicity_map.png';
    
plt.savefig(zmapfile,dpi=300)

plt.close()

# t50

fig, ax = plt.subplots(1,1)

fig.tight_layout()

# plotting UV photometry as background

ax.imshow(f090w_phot, norm=LogNorm(vmin=0.), origin='lower', cmap='cmr.sunburst')

# putting in the regions

patch_list, artist_list = reg_shape_list.get_mpl_patches_texts()

for p in patch_list:
    ax.add_patch(p)
for t in artist_list:
    ax.add_artist(t)

# just making little dots with plt.scatter for the clumps because that's wayyyy simpler than plotting ellipses

colormask = np.max(agebin_interp)-df_sfh.t_median_sfr

plt.scatter(clumpx,clumpy,c=colormask,cmap='cmr.lavender',s=50,alpha=0.95)
cb=plt.colorbar(shrink=0.57)
cb.set_ticks(ticks=np.round(np.linspace(np.min(colormask),np.max(colormask),num=6),decimals=2))
cb.set_label(r'$t_{50}$ (Gyr)',size=24)
ax.set_title(r'SGAS 1527 $t_{50}$',size=26)
    
ax.axis('off')

# adding labels for the clump indices so we know which one is which

texts = [ax.text(clumpx[i], clumpy[i], '%s' %i, ha='center', va='center', fontsize=7) for i in range(len(reg_array))]
adjust_text(texts, expand=(1.2, 2), arrowprops=dict(arrowstyle="-", color='k', lw=0.8))

ax.set_xlim(40,350)
ax.set_ylim(100,240)

t50mapfile = outdir+'1527clumpt_50_map.png'
    
plt.savefig(t50mapfile,dpi=300)

# mass versus t_50 plot with metallicity as cb

fig, ax = plt.subplots(1,1)

fig.tight_layout()

# just making little dots with plt.scatter for the clumps because that's wayyyy simpler than plotting ellipses

mass = df_sfh.logmass
t_50 = np.max(agebin_interp)-df_sfh.t_median_sfr
colormask = df_sfh.logzsol

plt.scatter(t_50,mass,c=colormask,cmap='cmr.lavender',s=50,alpha=0.95)
cb=plt.colorbar(shrink=0.9)
cb.set_ticks(ticks=np.round(np.linspace(np.min(colormask),np.max(colormask),num=6),decimals=2))
cb.set_label(r'$\log{\frac{Z}{Z_{\odot}}}$',size=24)
ax.set_title(r'SGAS 1527 Mass versus $t_{50}$',size=26)

# adding labels for the clump indices so we know which one is which
#texts = [ax.text(t_50[i],mass[i],'%s' %i, ha='center', va='center', fontsize=7) for i in range(len(mass))]
#adjust_text(texts, expand=(1.2, 2), arrowprops=dict(arrowstyle="-", color='k', lw=0.8))
ax.set_xlabel(r'$t_{50}$ (Gyr)')
ax.set_ylabel(r'$\log{\frac{M}{M_{\odot}}}$')
ax.set_xlim(0.,2.4)
ax.set_ylim(7.,10.)

massvt50file = outdir+'/1527massvt_50_map.png'
    
plt.savefig(massvt50file,dpi=300)

plt.close()


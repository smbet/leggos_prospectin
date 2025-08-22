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
from prospect.sources import CSPSpecBasis
from prospect.likelihood import lnlike_spec, lnlike_phot
from prospect.likelihood import chi_spec, chi_phot
from prospect.io import write_results as writer
from prospect.utils.obsutils import fix_obs
from prospect.models import SedModel
from prospect.fitting import fit_model
from prospect.plotting.utils import sample_posterior

from multiprocessing import Pool
from contextlib import closing
import pickle

import pandas as pd
import sedpy
from astropy.io import ascii
from astropy.io import fits
import gc
import decimal

from sedpy import observate

import prospect
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
#from prospect.plotting.utils import sample_posterior
from prospect.sources import CSPSpecBasis
from prospect.models import transforms
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

#from leggos_prospectin.lp_methods import*

### importing arguments first:
parser = argparse.ArgumentParser()
parser.add_argument("redshift", type=float, help="object redshift")
parser.add_argument("input_mag_path", type=str, help="input magnitude fits filepath")
parser.add_argument("out_path", type=str, help="h5 and corner plot output filepath")
parser.add_argument("--starting_index", type=int, default=0, help="clump index you want to start at")
#parser.add_argument("--filterlist", type=list, 
#    default=['jwst_f090w','jwst_f162m','jwst_f210m','jwst_f277w','jwst_f300m','jwst_f444w'],
#    help="JWST filters that you have magnitudes of")
parser.add_argument("--mag_units", type=str, 
    help="magnitude and error units, currently not supported sry they need to be microjansky")

args = parser.parse_args()

zred = np.float64(args.redshift)
in_fits = args.input_mag_path
outdir = args.out_path
h5path = args.out_path
ii_0 = args.starting_index

serial = True
poly_calibration=1
add_neb=True
snr = 5
verbose = False
counter_sps = 0

### super important bit this is where we select what kind of star formation history we want
### and right now it's set for a binned, or non-parametric model
model_params = TemplateLibrary["continuity_sfh"]
###

#########
# defining the methods that should be in an another script but aren't rn

# idk what this one is for
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

# some statistics thing
def gen_kde(zred_all, kde_factor, xrange_fac):
    #x1 = np.linspace(np.min(np.array(zred_all))-xrange_fac,np.max(np.array(zred_all)+xrange_fac),100) 
    #x1 = np.linspace(np.percentile(np.array(zred_all),1),np.percentile(np.array(zred_all),99),100) 
    sig1 = np.percentile(np.array(zred_all),50) - np.percentile(np.array(zred_all),16)
    sig2 = np.percentile(np.array(zred_all),84) - np.percentile(np.array(zred_all),50)
    jsig = np.sqrt((sig1**2) + (sig2**2))
    med = np.percentile(np.array(zred_all),50)
    x1 = np.linspace(med - (3*jsig), med + (3*jsig), 1000) 

    abc1 = scipy.stats.gaussian_kde(zred_all,kde_factor)
    evaluated = abc1.evaluate(x1)
    
    return x1,evaluated

# next two kind of say what they do in their names (kind of anyway)
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
    print(mags)
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
def zred_to_agebins(zred=None,agebins=None,nbins_sfh=7,**extras):
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

model_params = TemplateLibrary["continuity_sfh"]

def build_model(model_params = model_params, add_neb=True):

    
    #######Initial Age Bin Definitions########
    tuniv_yr = cosmo.age(zred).value*1e9
    nbins_sfh = 8
    tbinmax = (tuniv_yr*0.8)
    agelims = [0.0,7.4772] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv_yr)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    agebins = agebins.T
    print(10**agebins/1e9)
    ##########################################

    #initial parameters and whether they're free or fixed and also priors

    
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

#########

#---------------------------------------------------------------------------------------------------
#
# grabbing the file:

home = expanduser('~')
try:
    path = os.path.join(home,'**',in_fits)
except:
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
run_params['object_redshift'] = zred
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

# for loop where everything happens
list_of_spec_lists = []
for ii in range(len(clumpdf)-ii_0):
    
    run_params['clump_ind']=ii+ii_0

    obs = load_obs(**run_params)
    sps = load_sps(**run_params)
    model_params = build_model()

    model = PolySpecModel(model_params)
    theta = model.theta.copy()
    
    #initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)
    #title_text = ','.join(["{}={}".format(p, model.params[p][0]) 
                           #for p in model.free_params])
    
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

    output = fit_model(obs, model, sps,**run_params)
    print('Success! Done fitting in {0}s'.format(output["sampling"][1]))

    pdur = None
    edur = None

    run_params["outfile"] = '1527clump_'+str(run_params['clump_ind'])#+'_'+str(model_params['agebins'])+'agebins'
    outroot = "{0}_{1}".format(run_params['outfile'], int(time.time()))
    
    hfile = h5path + '/' + outroot + '_dynesty.h5'
    
    print("Writing to file {}".format(hfile))
    writer.write_hdf5(hfile, run_params, model, obs,
                  output["sampling"][0], output["optimization"][0],
                  tsample=output["sampling"][1],
                  toptimize=output["optimization"][1])

    result, obs, _ = reader.results_from(hfile, dangerous=False)

    nparam = len(model.theta)

    start_num = 3
    nparam = len(model.theta)
    imax = np.argmax(result['lnprobability'])
    #Dynesty
    theta_max = result["chain"][imax, :]
    thin = 1
    
    print('MAP value: {}'.format(theta_max))
    
    cornerfig = reader.subcorner(result, start=start_num, thin=thin, truths=theta_max, 
                            fig=plt.subplots(nparam,nparam,figsize=(27,27))[0],color = 'black')
    plt.savefig(outdir+'/cornerplot'+outroot+'.png',dpi=300)
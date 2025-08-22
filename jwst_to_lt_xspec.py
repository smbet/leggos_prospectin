import time, sys, os
import h5py
import numpy as np
import scipy

import numpy as np
import time
import os

from glob import glob
from os.path import expanduser
from scipy.stats import linregress
from astropy.table import Table, vstack
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from astropy.io import fits
import pandas as pd
import argparse

'''
super straightforward script for turning the wavelength units of a jwst spectrum from um to angstrom,
although actually the input file will just have indices you have to look and see what wavelengths
in um those indices correspond to
'''

### importing arguments first:
parser = argparse.ArgumentParser()
parser.add_argument("input_fits_path", type=str, help="input 1d spectrum fits file filepath")
parser.add_argument("output_fits_path", type=str, help="output 1d spectrum fits file filepath")
parser.add_argument("min_wavelength", type=float, help="min wavelength in microns")
parser.add_argument("max_wavelength", type=float, help="max wavelength in microns")
args = parser.parse_args()

outname=args.output_fits_path
micron_wavelengths = [args.min_wavelength, args.max_wavelength]


### getting jwst 1d spectrum
fitspath = args.input_fits_path
fitspath = glob(fitspath)
hdu = fits.open(fitspath[0])
dat = hdu[0].data.copy()
head = hdu[0].header.copy()
hdu.close()
# making table from fits file
dat_thing = np.array([dat[ii] for ii in range(len(dat))])
dat_table = pd.DataFrame(dat_thing,columns=['flux'])

### converting the wavelengths 
wave_ind = np.array(list(range(len(dat))))+1 # not 0 based
x = [wave_ind[0],wave_ind[-1]]
y = [micron_wavelengths[0],micron_wavelengths[1]]
line = linregress(x,y)
m,b = line.slope,line.intercept
microns=(wave_ind*m)+b
angstrom = 10000*microns
dat_table['wavelength'] = angstrom

### saving new wavelength and flux table to a new fits file
table=Table([dat_table.wavelength,dat_table.flux],names=['wavelength','flux'])
table.write(outname,format='fits',overwrite=True)
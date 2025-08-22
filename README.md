Here are steps for setting up a conda environment and going from some 
input magnitudes to a fitted SED with star formation histories and plots
with this prospector-based code. This is just the bare bones version, 
for more info there's also a readthedocs.

First create a conda environment and install all the basic libraries like astropy, etc. using the following command:

conda create --name prospectin --file spec_list.txt

Then activate the environment:

conda activate prospectin

Next we need to install a bunch of more specialized packages using pip, so next enter:

pip install adjusttext==1.3.0 astro-prospector==1.4.0 cmasher==1.8.0 contourpy==1.3.0 corner dynesty==2.1.5 emcee==3.1.6 fsps h5py==3.12.1 pyogrio==0.10.0 pyparsing==3.2.1 pyproj==3.6.1 pyregion regions==0.8

And then:

python -m pip install astro-sedpy

My version info, for reference in case something goes wrong with the automatic setup from the file:

Name                    Version                   Build  Channel
adjusttext                1.3.0                    pypi_0    pypi
arrow                     1.3.0              pyhd8ed1ab_1    conda-forge
astro-prospector          1.4.0                    pypi_0    pypi
astropy                   6.0.1            py39h373d45f_0    conda-forge
cmasher                   1.8.0                    pypi_0    pypi
contourpy                 1.3.0                    pypi_0    pypi
corner                    2.2.4.dev2+g6fba5cd          pypi_0    pypi
dynesty                   2.1.5                    pypi_0    pypi
emcee                     3.1.6                    pypi_0    pypi
fsps                      0.4.8.dev36+g0150b9b          pypi_0    pypi
h5py                      3.12.1                   pypi_0    pypi
matplotlib                3.9.4                    pypi_0    pypi
matplotlib-inline         0.1.7              pyhd8ed1ab_1    conda-forge
numpy                     1.26.4           py39h7aa2656_0    conda-forge
pandas                    2.2.3                    pypi_0    pypi
pip                       24.3.1             pyh8b19718_2    conda-forge
pyogrio                   0.10.0                   pypi_0    pypi
pyparsing                 3.2.1                    pypi_0    pypi
pyproj                    3.6.1                    pypi_0    pypi
pyregion                  2.2.0                    pypi_0    pypi
python                    3.9.21          h5f1b60f_1_cpython    conda-forge
regions                   0.8                      pypi_0    pypi
scipy                     1.13.1           py39h3d5391c_0    conda-forge

Next run this in your python-fsps directory:

export SPS_HOME=$(pwd)/src/fsps/libfsps

It's a command that needs to be run in order for FSPS to work (python-fsps just being
a pythony way to access the fsps code, which is in fortran). You'll need to run that command in the python-fsps
directory whenever you start a new session in your terminal.

Now we're ready to take prospector and make an SED fit to a set of input magnitudes for a given object
or list of objects.

We'll do this with the prospectin_fits.py script, which requires the following arguments:

redshift, type=float, galaxy redshift
input_mag_path, type=str, input magnitudes fits filepath, example:
"./1527/eye_fit_phot/clumpFluxes1527_NRConly.fits"
input_h5_path, type=str, input h5 directory filepath (the directory where
 the files with the baysian sed fits that prosector made are stored), example:
"./1527/michael_clumps/hfives/"
optional arguments:
--clump_index, type=int, index of object in the table that you want prospector
to start on. Useful to set it to the length of the table if you're messing around with the parameters
and don't want to fit every clump every time you run the script

So the full completed text entered into the command line will look something like:

python prospectin_fits.py 2.763 "./1527/eye_fit_phot/clumpFluxes1527_NRConly.fits" "./1527/michael_clumps/hfives/"

Important Note: right now unless the filters are the same 6 from JWST and are named exactly like this
in the magnitude table then it won't work (sorry):
F090W_med,F090err,F162_med,F162err, and so on for the other 4 filters in the set.

The initial values and priors are set to fit lensed clumps, if you're looking at
whole galaxies or something you may need to go into the code to change some of them, 
they can be found in the method called "build_model".

This script will be doing a bunch of baysian statistics to get you your fits and will take a long time for each object that your input table contains.

Once prospectin_fits has been run, for each object in your input magnitudes table, in your output directory
you'll have an h5 file with prospector's fit saved to it, as well as a corner plot of all the free parameters
which you can look at to inspect how well your fit has converged.

Now that you've made your fits that you are hopefully happy with, you can use
them to make some pretty plots to visualize the data.
For this we're going to use prospectin_plots.py. For prospectin_plots you'll need the following arguments:

redshift, type=float, galaxy redshift
input_mag_path, type=str, input magnitudes fits filepath, example:
"./1527/eye_fit_phot/clumpFluxes1527_NRConly.fits"
input_h5_path, type=str, input h5 directory filepath (the directory where
 the files with the baysian sed fits that prosector made are stored), example:
"./1527/hfives/"
input_image_path, type=str, filepath for fits image of objects that were observed, presumably one of the
ones that magnitudes were pulled from, example:
"/1527/S1527_F090W_noBCG.fits"
reg_path, type=str, regions filepath so you can have the regions that were used to extract photometry shown
overlaid on the image you gave it in input_image_path, example:
"/1527/clumpRegions_F090W_F444W.reg"
out_path, type=str, directory where you want the many plots that this script will make to be stored, example:
"/exampleoutdir/"
optional arguments:
--clump_index, type=int, index of object in the table that you want the clump-by-clump plots to start on. As in the case
of prospectin_fits this is useful when you're in the code fiddling with things (like the layout of the plots for example)

So the full completed text entered into the command line will look something like:

python prospectin_plots.py 2.762 "/1527/eye_fit_phot/clumpFluxes1527_NRConly.fits" "/1527/hfives/" "/1527/S1527_F090W_noBCG.fits" "/1527/clumpRegions_F090W_F444W.reg" "/testoutdir/"

Once finished, the prospectin_plots script outputs a plethora of plots. Specifically:

For each individual object:
1. A big star-formation history plot.
2. A best fit model plot.
3. Both of those plots together.
4. SFH plot with model plot inset.

Then for all the objects together:
1. A plot of the regions you extracted magnitudes from overlaid on a fits image of the field.
2. Plot of the regions overlaid on fits image of field but the regions are colormapped to (lensing-uncorrected) mass.
3. Plot of the regions overlaid on fits image of field but the regions are colormapped to metallicity.
4. Plot of the regions overlaid on fits image of field but the regions are colormapped to t_50.
5. Scatter plot of mass versus t_50 and colormapped to metallicity.
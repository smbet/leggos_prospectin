from setuptools import setup

import os
import glob

def readfile(filename):
    with open(filename, 'r+') as f:
        return f.read()

# taking some of Matt's code from vetrr:

def get_scripts():
    """ Grab all the scripts in the bin directory.  """
    scripts = []
    if os.path.isdir('bin'):
        scripts = [fname for fname in glob.glob(os.path.join('bin', '*'))
                   if not os.path.basename(fname).endswith('.rst')]
    return scripts


scripts = get_scripts()

setup(name='leggos_prospectin',
      version='0.1',
      description='Software for prospectin (galaxy) clumps.',
      long_description=readfile('README.md'),
      url='https://github.com/smbet/leggos_prospectin',
      author='Sierra Bet',
      author_email="sbet@uw.edu",
      packages=['prospectinclumps'],
      license=readfile('LICENSE'),
      scripts=scripts,
    )
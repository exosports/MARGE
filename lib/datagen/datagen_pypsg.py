"""
Contains functions related to generating/processing data to be used for 
training an NN with MARGE.

generate_data: Generates data according to specified parameters.

process_data: Processed generated data.

"""

import sys, os
import glob
import configparser
import numpy as np


def generate_data(cfile, data_dir=None):
    """
    Handles data generation according to specified parameters.
    """
    print('SmithEtal is not configured to generate data.')
    print('Set datagen=False and try again.')
    sys.exit(1)


def process_data(cfile, data_dir, preserve=True):
    """
    Handles data processing to match what MARGE expects.
    NOTE: this will nearly double the amount of disk space used if 
          the `preserve` flag is True!
    """
    print('Processing the pypsg data...')
    # Load config file
    config = configparser.ConfigParser(allow_no_value=True)
    config.read_file(open(cfile, 'r'))
    for section in config:
        if section != "DEFAULT":
            conf = config[section]
            ### Unpack the variables ###
            # Directory with unprocessed data
            if not os.path.isabs(data_loc):
                raise Exception('For data processing with pypsg, data_loc '\
                              + 'must be an absolute path.')
            train_loc = os.path.join(data_loc, 'train', '')
            valid_loc = os.path.join(data_loc, 'valid', '')
            test_loc  = os.path.join(data_loc, 'test' , '')
            # Set up directory tree
            if not os.path.isabs(data_dir):
                raise Exception('For data processing with pypsg, data_dir '\
                              + 'must be an absolute path.')

            train_dir = os.path.join(data_dir, 'train', '')
            valid_dir = os.path.join(data_dir, 'valid', '')
            test_dir  = os.path.join(data_dir, 'test' , '')
            # Get relevent indices for slicing
            ibeg = conf.getint('ibeg')
            iend = conf.getint('iend')
            xbeg = conf.getint('xbeg')
            xend = conf.getint('xend')
            obeg = conf.getint('obeg')
            oend = conf.getint('oend')

            # Process the data
            # Hardcoded 3: training, validation, and testing
            for i in range(3):
                if i==0:
                    indir  = train_loc
                    outdir = train_dir
                    print('  Training data:')
                elif i==1:
                    indir  = valid_loc
                    outdir = valid_dir
                    print('  Validation data:')
                else:
                    indir  = test_loc
                    outdir = test_dir
                    print('  Testing data:')
                # Load the data file names
                foos = glob.glob(indir + '*.npy')
                # Process each file
                for j in range(len(foos)):
                    print('  File '+str(j+1)+'/'+str(len(foos)), end='\r')
                    dat     = np.load(foos[j])
                    inarr   = dat[:, ibeg:iend]
                    outarr  = dat[:, obeg:oend]
                    savearr = np.concatenate((inarr, outarr), axis=-1)
                    fsave   = outdir + foos[j].rsplit(os.sep, 1)[-1]
                    np.save(fsave, savearr)
                print('')

            # Slice and save the wavelength grid
            fxvals = conf['xvals']
            if not os.path.isabs(fxvals):
                fxvals = os.path.join(cfile.rsplit(os.sep, 1)[0], fxvals)
            xvals  = dat[0, xbeg:xend]
            np.save(fxvals, xvals)
            print("The pypsg data has been processed.")

    return

















"""
Contains functions related to generating/processing data to be used for 
training an NN with MARGE.

generate_data: Generates data according to specified parameters.

process_data: Processed generated data.

"""

import sys, os
import subprocess
import configparser
import numpy as np
import scipy.constants as sc

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + \
                '/../../modules/BART/code/')
import constants as const


def generate_data(cfile, data_dir=None):
    """
    Handles data generation according to specified parameters.
    """
    out = subprocess.run([os.path.dirname(os.path.abspath(__file__)) \
                          + '/../../modules/BART/BART.py', '-c', cfile] )
    return


def process_data(cfile, data_dir, preserve=True):
    """
    Handles data processing to match what MARGE expects.
    NOTE: this will nearly double the amount of disk space used if 
          the `preserve` flag is True!
    """
    print('Processing the BART data...')
    # Load config file
    config = configparser.ConfigParser(allow_no_value=True)
    config.read_file(open(cfile, 'r'))
    for section in config:
        if section != "DEFAULT":
            conf = config[section]
            ### Unpack the variables ###
            # Output directory
            loc_dir = os.path.join(conf['loc_dir'], '')
            if not os.path.isabs(loc_dir):
                loc_dir = os.path.join(
                            os.path.abspath(cfile).rsplit('/', 1)[0], 
                            loc_dir)
            fparams = loc_dir + 'output.npy'
            # Saved models
            savemodel = conf['savemodel']
            # Model directory
            model_dir = loc_dir + savemodel.replace('.npy', '') + '/'
            # Load file names w/ absolute paths
            fmodels   = os.listdir(model_dir)
            fmodels   = [os.path.join(model_dir, foo) for foo in fmodels]
            # Set up directory tree
            if not os.path.isabs(loc_dir):
                data_dir = os.path.join(loc_dir, data_dir, '')
            else:
                data_dir = os.path.join(data_dir, '')
            train_dir = data_dir + 'train/'
            valid_dir = data_dir + 'valid/'
            test_dir  = data_dir + 'test/'

            # Load params file, stack the chains, reshape to (Niter, params)
            modelper = conf.getint('modelper')
            pars   = np.load(fparams)
            params = np.zeros((pars.shape[0]*pars.shape[2], pars.shape[1]))
            cbeg   = 0 # counters for slicing params
            cend   = 0
            for i in range(int(np.ceil(pars.shape[-1]/modelper))):
                ibeg =      i   *modelper
                iend = min((i+1)*modelper, pars.shape[-1])
                for c in range(pars.shape[0]):
                        cend += iend-ibeg
                        params[cbeg:cend] = pars[c,:,ibeg:iend].T.copy()
                        cbeg += iend-ibeg
            del pars # Free up memory (probably minor)

            # Transform mass (Mjup) -> surface gravity (cm s-2)
            params[:, 6] = 100 * sc.G * params[:, 6] * const.Mjup \
                           / (params[:, 5] * 1000)**2 # Radius km --> m

            # Process the data
            nfoos = len(fmodels)
            ntest = max(np.floor(nfoos * 0.1), 1)
            nval  = max(np.floor(nfoos * 0.2), 1)
            ntr   = max(nfoos - ntest - nval,  1)
            nproc = 0 # Counter
            for n, foo in enumerate(fmodels):
                print('  File '+str(n+1)+'/'+str(len(fmodels)), end='\r')
                dat   = np.load(foo) # Load data file
                # Stack the chains
                stack = np.zeros((dat.shape[0]*dat.shape[2], dat.shape[1]))
                for c in range(dat.shape[0]):
                    cbeg = modelper *  c
                    cend = modelper * (c+1)
                    stack[cbeg:cend] = dat[c].T.copy()
                # Remove empty part of `stack` array (0s)
                badinds = np.where(np.all(stack == 0, axis=-1))[0]
                stack   = np.delete(stack, badinds, axis=0)
                # Slice relevant position of `params`
                pslice  = params[nproc : nproc + stack.shape[0]].copy()
                nproc  += stack.shape[0]
                # Remove any data vectors that are all -1s (out of bounds)
                badinds = np.where(np.all(stack == -1, axis=-1))[0]
                stack   = np.delete(stack , badinds, axis=0)
                pslice  = np.delete(pslice, badinds, axis=0)
                # Combine arrays so each vector is params then model
                savearr = np.concatenate((pslice, stack), axis=-1)
                del pslice, stack
                # Save the data
                fsave   = savemodel.replace('.npy', 
                                     str(n).zfill(len(str(len(fmodels)))) \
                                     + '.npy')
                if   n < ntr:
                    np.save(train_dir + fsave, savearr)
                elif n < ntr + nval:
                    np.save(valid_dir + fsave, savearr)
                elif n < ntr + nval + ntest:
                    np.save(test_dir  + fsave, savearr)
                del savearr

                if not preserve:
                    out = subprocess.run(['rm', '-f', foo])
            print('')

    print('\nThe generated data has been processed and split into ')
    print('training, validation, and test sets.')
    print('Note that ~70% of the data is used for training, ' + \
          '~20% for validation, ')
    print('and ~10% for testing, based on the # of files created ' + \
          'during data generation.')
    print('\nIf few data files were created during data generation, ' + \
          'the data subsets ')
    print('may not reflect this split percentage.')
    print('If this occurs, the user must manually adjust the subsets.')

    return

















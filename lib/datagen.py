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

import utils as U


def generate_data(code, cfile=None, data_dir=None):
    """
    Handles data generation according to specified parameters.
    """
    if code == 'BART':
        out = subprocess.run([os.path.dirname(os.path.abspath(__file__)) \
                              + '/../modules/BART/BART.py', '-c', cfile] )
        return
    else:
        print('The code specified is not currently implemented.')
        print('Please add this to lib/datagen.py, and submit a pull request.')
        sys.exit()


def process_data(code, cfile, data_dir, preserve=True):
    """
    Handles data processing to match what MARGE expects.
    NOTE: this will nearly double the amount of disk space used if 
          the `preserve` flag is True!
    """
    print('Processing the generated data...')
    if code == 'BART':
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
                                '..', loc_dir)
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
                U.make_dir(data_dir)
                U.make_dir(train_dir)
                U.make_dir(valid_dir)
                U.make_dir(test_dir)

                # Load params file, stack the chains, reshape to (Niter, params)
                modelper = conf.getint('modelper')
                pars   = np.load(fparams)
                params = np.zeros((pars.shape[0]*pars.shape[2], pars.shape[1]))
                for i in range(pars.shape[-1]//modelper):
                    ibeg =  i   *modelper
                    iend = (i+1)*modelper
                    for c in range(pars.shape[0]):
                            cbeg = modelper * (i*pars.shape[0] +  c   )
                            cend = modelper * (i*pars.shape[0] + (c+1))
                            params[cbeg:cend] = pars[c,:,ibeg:iend].T.copy()
                del pars # Free up memory (probably minor)

                # Process the data
                nfoos = len(fmodels)
                ntest = max(np.floor(nfoos * 0.1), 1)
                nval  = max(np.floor(nfoos * 0.2), 1)
                ntr   = nfoos - ntest - nval
                nproc = 0 # Counter
                for n, foo in enumerate(fmodels):
                    print('File '+str(n+1)+'/'+str(len(fmodels)), end='\r')
                    dat   = np.load(foo) # Load data file
                    stack = dat[0]       # Stack the chains
                    for c in range(1, dat.shape[0]):
                        stack = np.hstack((stack, dat[c]))
                    del dat              # Free up memory
                    stack = stack.T      # Reshape so data vectors are axis 0
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
    elif code == 'PSG':
        out_dir = os.path.join(data_dir, 'proc', '')
        U.make_dir(out_dir)
        U.make_dir(out_dir + 'train/')
        U.make_dir(out_dir + 'valid/')
        U.make_dir(out_dir + 'test/')
        ftrain, fvalid, ftest = U.get_file_names(data_dir)

        allfoo = np.concatenate([ftrain, fvalid, ftest])

        for n, foo in enumerate(allfoo):
            dat = np.load(foo)
            xx  = dat[:, 0:28]
            xx[:, 14:14+12] = np.log10(xx[:, 14:14+12])
            yy  = dat[:, 15374:15374+4379]
            savearr = np.concatenate([xx, yy], axis=-1)
            np.save(foo.replace(data_dir, out_dir), savearr)
    else:
        print('The code specified is not currently implemented.')
        print('Please add this to lib/datagen.py, and submit a pull request.')
        sys.exit()

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

















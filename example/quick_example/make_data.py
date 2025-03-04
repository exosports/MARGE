"""
This script creates a dataset of angles theta and the corresponding (x, y) 
position on a unit circle.
"""

import sys, os
import numpy as np


def main(seed=0, 
         nfrac_train=0.7, nfrac_valid=0.15, 
         datadir='data', 
         nbatch_save=64):
    # Set random seed for reproducibility
    np.random.seed(seed)
    # Sample the angles between 0 and 2pi
    thetalo, thetahi = 0, 2*np.pi
    nsamp            = int(1e5)
    inputs      = np.zeros((nsamp, 1))
    inputs[:,0] = np.random.uniform(thetalo, thetahi, nsamp)
    # Calculate the (x,y) positions for each angle
    outputs = np.hstack([np.cos(inputs), np.sin(inputs)])
    
    # Split the data into training, validation, and test sets
    ntr  = int(nsamp * nfrac_train)
    nval = int(nsamp * nfrac_valid)
    nte  = nsamp - ntr - nval

    inputs_tr  = inputs [:ntr]
    outputs_tr = outputs[:ntr]
    
    # Normalize the data by the training data mean and standard deviation
    inputs_tr_mean  = np.mean(inputs_tr, axis=0)
    inputs_tr_stdev = np.std (inputs_tr, axis=0)

    outputs_tr_mean  = np.mean(outputs_tr, axis=0)
    outputs_tr_stdev = np.std (outputs_tr, axis=0)
    
    inputs_norm  = (inputs  - inputs_tr_mean ) / inputs_tr_stdev
    outputs_norm = (outputs - outputs_tr_mean) / outputs_tr_stdev

    inputs_norm   = (inputs   - inputs_tr_mean ) / inputs_tr_stdev
    outputs_norm  = (outputs  - outputs_tr_mean) / outputs_tr_stdev

    # Save the data
    traindir = os.path.join(datadir, 'train')
    validdir = os.path.join(datadir, 'valid')
    testdir  = os.path.join(datadir, 'test' )
    os.makedirs(traindir, exist_ok=True)
    os.makedirs(validdir, exist_ok=True)
    os.makedirs(testdir , exist_ok=True)
    savedirs = [traindir, validdir, testdir]
    nsave = [int(np.ceil(ntr/nbatch_save)), int(np.ceil(nval/nbatch_save)), 
             int(np.ceil(nte/nbatch_save))]
    offset = [0, ntr, ntr+nval, ntr+nval+nte]
    
    for ind in range(len(savedirs)):
        for i in range(nsave[ind]):
            np.savez(os.path.join(savedirs[ind], str(i).zfill(4)), 
                     x=inputs_norm [offset[ind] + i*nbatch_save : min(offset[ind+1], offset[ind] + (i+1)*nbatch_save)], 
                     y=outputs_norm[offset[ind] + i*nbatch_save : min(offset[ind+1], offset[ind] + (i+1)*nbatch_save)])
    
    return

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
'''
Driver for MARGE
'''


import configparser
import sys, os
import numpy as np

import keras
from keras import backend as K

libdir = os.path.dirname(__file__) + '/lib'
sys.path.append(libdir)

import datagen as D
import loader  as L
import NN
import stats   as S
import utils   as U


def MARGE(confile):
    """
    Main driver for the software.
    For assistance, consult the MARGE User Manual.

    Inputs
    ------
    confile : path/to/configuration file.

    Examples
    --------
    See config.cfg in the top-level directory for example.
    Run it from a terminal like
      user@machine:/dir/to/MARGE$ ./MARGE.py config.cfg
    """
    # Load configuration file
    config = configparser.ConfigParser(allow_no_value=True)
    config.read_file(open(confile, 'r'))

    # Run everything specified in config file
    for section in config:
        if section != "DEFAULT":
            conf = config[section]
            ### Unpack the variables ###
            # Top-level params
            datagen     = conf.getboolean("datagen")
            code        = conf["code"]
            cfile       = conf["cfile"]
            processdat  = conf.getboolean("processdat")
            preservedat = conf.getboolean("preservedat")
            NNmodel     = conf.getboolean("NNmodel")
            trainflag   = conf.getboolean("trainflag")
            validflag   = conf.getboolean("validflag")
            testflag    = conf.getboolean("testflag")
            resume      = conf.getboolean("resume")
            TFRfile     = conf["TFR_file"]
            if TFRfile != '':
                TFRfile = TFRfile + '_' # Separator for file names
            buffer_size = conf.getint("buffer")
            ncores      = conf.getint("ncores")
            if  ncores  > os.cpu_count():
                ncores  = os.cpu_count()
            normalize   = conf.getboolean("normalize")
            scale       = conf.getboolean("scale")
            seed        = conf.getint("seed")

            # Directories
            inputdir  = os.path.abspath(conf["inputdir" ]) + '/'
            outputdir = os.path.abspath(conf["outputdir"]) + '/'
            if not os.path.isabs(conf["plotdir"]):
                plotdir = os.path.join(outputdir, conf["plotdir"], '')
            else:
                plotdir = os.path.join(           conf["plotdir"], '')
            if not os.path.isabs(conf["datadir"]):
                datadir = os.path.join(outputdir, conf["datadir"], '')
            else:
                datadir = os.path.join(           conf["datadir"], '')
            if not os.path.isabs(conf["preddir"]):
                preddir = os.path.join(outputdir, conf["preddir"], '')
            else:
                preddir = os.path.join(           conf["preddir"], '')
            
            # Create the directories if they do not exist
            U.make_dir(inputdir)
            U.make_dir(inputdir+'TFRecords/')
            U.make_dir(outputdir)
            U.make_dir(plotdir)
            U.make_dir(datadir)
            U.make_dir(preddir)
            U.make_dir(preddir+'valid/')
            U.make_dir(preddir+'test/')

            # Files to save
            fmean         = conf["fmean"]
            fstdev        = conf["fstdev"]
            fmin          = conf["fmin"]
            fmax          = conf["fmax"]
            fsize         = conf["fsize"]
            rmse_file     = conf["rmse_file"]
            r2_file       = conf["r2_file"]

            # Data info
            inD         = conf.getint("input_dim")
            outD        = conf.getint("output_dim")
            if scale:
                scalelims = [int(num) for num in conf["scalelims"].split(',')]
            else:
                scalelims = [0., 1.] # Won't change the data values

            # Model info
            weight_file   = outputdir + conf["weight_file"]
            epochs        = conf.getint("epochs")
            batch_size    = conf.getint("batch_size")
            patience      = conf.getint("patience")
            prelayers     = conf["prelayers"]
            if prelayers != "None" and prelayers != "":
                prelayers =  [int(num) for num in prelayers.split()]
            else:
                prelayers = None
            layers        =  [int(num) for num in conf["layers"].split()]
            ilog          = conf.getboolean("ilog")
            olog          = conf.getboolean("olog")

            # Learning rate parameters
            lengthscale   = conf.getfloat("lengthscale")
            max_lr        = conf.getfloat("max_lr")
            clr_mode      = conf["clr_mode"]
            clr_steps     = conf["clr_steps"]

            # Plotting parameters
            xlabel        = conf["xlabel"]
            xvals         = np.load(inputdir + conf["xvals"])
            ylabel        = conf["ylabel"]
            plot_cases    = [int(num) for num in conf["plot_cases"].split()]

            # Generate data set
            if datagen:
                print('\nMode: Generate data\n')
                D.generate_data(code, inputdir+cfile)

            if processdat:
                print('\nMode: Process data\n')
                D.process_data(code, inputdir+cfile, datadir, preservedat)

            # Train a model
            if NNmodel:
                print('\nMode: Neural network model\n')
                NN.driver(inputdir, outputdir, datadir, plotdir, preddir, 
                          trainflag, validflag, testflag, 
                          normalize, fmean, fstdev, 
                          scale, fmin, fmax, scalelims, 
                          fsize, rmse_file, r2_file, 
                          inD, outD, ilog, olog, 
                          TFRfile, batch_size, ncores, buffer_size, 
                          prelayers, layers, 
                          lengthscale, max_lr, clr_mode, clr_steps, 
                          epochs, patience, weight_file, resume, 
                          plot_cases, xvals, xlabel, ylabel)

    return


if __name__ == "__main__":
    #U.limit_mem()
    try:
        MARGE(*sys.argv[1:])
    except MemoryError:
        sys.stderr.write('\nERROR: Memory limit exceeded.')
        sys.exit(1)






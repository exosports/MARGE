#!/usr/bin/env python3
'''
Driver for MARGE
'''

import sys, os
import configparser
import importlib
import numpy as np

import keras
from keras import backend as K

libdir = os.path.join(os.path.dirname(__file__), 'lib', '')
sys.path.append(libdir)

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
            # Directories
            inputdir  = os.path.join(os.path.abspath(conf["inputdir" ]), '')
            outputdir = os.path.join(os.path.abspath(conf["outputdir"]), '')
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
            U.make_dir(os.path.join(inputdir, 'TFRecords', ''))
            U.make_dir(outputdir)
            U.make_dir(plotdir)
            U.make_dir(datadir)
            U.make_dir(os.path.join(datadir, 'train', ''))
            U.make_dir(os.path.join(datadir, 'valid', ''))
            U.make_dir(os.path.join(datadir, 'test', ''))
            U.make_dir(preddir)
            U.make_dir(os.path.join(preddir, 'valid', ''))
            U.make_dir(os.path.join(preddir, 'test', ''))

            # Main options
            datagen     = conf.getboolean("datagen")
            cfile       = conf["cfile"]
            processdat  = conf.getboolean("processdat")
            preservedat = conf.getboolean("preservedat")
            NNmodel     = conf.getboolean("NNmodel")
            gridsearch  = conf.getboolean("gridsearch")
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

            # Import the datagen module
            if datagen or processdat:
                datagenfile = conf["datagenfile"].rsplit(os.sep, 1)
                if len(datagenfile) == 2:
                    if os.path.isabs(datagenfile[0]):
                        sys.path.append(datagenfile[0])
                    else:
                        sys.path.append(inputdir+datagenfile[0])
                else:
                    # Look in inputdir first, check lib/datagen/ after
                    sys.path.append(inputdir)          
                    sys.path.append(os.path.join(libdir, 'datagen', ''))
                D = importlib.import_module(datagenfile[-1])

            # Files to save
            fmean     = conf["fmean"]
            fstdev    = conf["fstdev"]
            fmin      = conf["fmin"]
            fmax      = conf["fmax"]
            fsize     = conf["fsize"]
            rmse_file = conf["rmse_file"]
            r2_file   = conf["r2_file"]

            # Data info
            inD  = conf.getint("input_dim")
            outD = conf.getint("output_dim")

            if conf["ilog"] in ["True", "true", "T", "False", "false", "F"]:
                ilog = conf.getboolean("ilog")
            elif conf["ilog"] in ["None", "none", ""]:
                ilog = False
            elif conf["ilog"].isdigit():
                ilog = int(conf["ilog"])
            elif any(pun in conf["ilog"] for pun in [",", " ", "\n"]):
                if "," in conf["ilog"]:
                    ilog = [int(num) for num in conf["ilog"].split(',')]
                else:
                    ilog = [int(num) for num in conf["ilog"].split()]
                if any(num >= inD for num in ilog):
                    raise ValueError("One or more ilog indices exceed the " + \
                                     "specified number of inputs.")
            else:
                raise ValueError("ilog specification not understood.")

            if conf["olog"] in ["True", "true", "T", "False", "false", "F"]:
                olog = conf.getboolean("olog")
            elif conf["olog"] in ["None", "none", ""]:
                olog = False
            elif conf["olog"].isdigit():
                olog = int(conf["olog"])
            elif any(pun in conf["olog"] for pun in [",", " ", "\n"]):
                if "," in conf["olog"]:
                    olog = [int(num) for num in conf["olog"].split(',')]
                else:
                    olog = [int(num) for num in conf["olog"].split()]
                if any(num >= outD for num in olog):
                    raise ValueError("One or more olog indices exceed the " + \
                                     "specified number of outputs.")
            else:
                raise ValueError("olog specification not understood.")

            if scale:
                scalelims = [int(num) for num in conf["scalelims"].split(',')]
            else:
                scalelims = [0., 1.] # Won't change the data values
            try:
                filters = conf["filters"].split()
                filt2um = float(conf["filt2um"])
                print('\nFilters specified. Will compute performance ' \
                    + 'metrics over integrated bandpasses.\n')
            except:
                filters = None
                filt2um = 1.
                print('\nFilters not specified. Will compute performance ' \
                    + 'metrics for each output.\n')

            # Model info
            if not os.path.isabs(conf["weight_file"]):
                weight_file = outputdir + conf["weight_file"]
            else:
                weight_file = conf["weight_file"]
            epochs      = conf.getint("epochs")
            batch_size  = conf.getint("batch_size")
            patience    = conf.getint("patience")
            if gridsearch:
                architectures = conf["architectures"].split('\n')
                layers        = [arch.split() 
                                 for arch in conf["layers"].split('\n')]
                lay_params    = [arch.split()
                                 for arch in conf["lay_params"].split('\n')]
                nodes         = [[int(num) for num in arch.split()] 
                                 for arch in conf["nodes"].split('\n')]
                activations   = [arch.split() 
                                 for arch in conf["activations"].split('\n')]
                act_params    = [arch.split()
                                 for arch in conf["act_params"].split('\n')]
                # Make sure the right number of entries exist
                if len(architectures) != len(layers):
                    raise Exception("Number of architecture names and sets " \
                                  + "of layers do not match.")
                elif len(architectures) != len(lay_params):
                    raise Exception("Number of architecture names and sets " \
                                  + "of layer parameters do not\nmatch.")
                elif len(architectures) != len(nodes):
                    raise Exception("Number of architecture names and sets " \
                                  + "of nodes do not match.")
                elif len(architectures) != len(activations):
                    raise Exception("Number of architecture names and sets " \
                                  + "of activations do not match.")
                elif len(architectures) != len(act_params):
                    raise Exception("Number of architecture names and sets " \
                                  + "of activation parameters do\nnot match.")
                for i in range(len(nodes)):
                    # Check for allowed layer types
                    if len(layers[i]) - layers[i].count("dense")     \
                                      - layers[i].count("conv1d")    \
                                      - layers[i].count("maxpool1d") \
                                      - layers[i].count("avgpool1d") \
                                      - layers[i].count("flatten") > 0:
                        raise Exception('Invalid layer type(s) specified. ' \
                                      + 'Allowed options: dense, conv1d,\n' \
                                      + 'maxpool1d, avgpool1d, flatten')
                    # Number of layers with nodes
                    nlay = layers[i].count("dense") + layers[i].count("conv1d")
                    if nlay != len(nodes[i]):
                        raise Exception("Number of Dense/Conv layers does "  \
                            + "not match the number of hidden\nlayers with " \
                            + "nodes.")
                    if len(layers[i]) != len(lay_params[i]):
                        raise Exception("Number of layer types does not " \
                            + "match the number of layer parameters.")
                    else:
                        # Set default layer parameters if needed
                        for j in range(len(layers[i])):
                            if lay_params[i][j] == 'None':
                                if layers[i][j] == 'conv1d':
                                    lay_params[i][j] = 3
                                elif layers[i][j] == 'maxpool1d' or \
                                     layers[i][j] == 'avgpool1d':
                                    lay_params[i][j] = 2
                            else:
                                lay_params[i][j] = int(lay_params[i][j])
                    if len(activations[i]) != len(nodes[i]):
                        raise Exception("Number of activation functions does " \
                            + "not match the number of hidden\nlayers with "   \
                            + "nodes.")
                    if len(activations[i]) != len(act_params[i]):
                        raise Exception("Number of activation functions does " \
                            + "not match the number of\nactivation function "  \
                            + "parameters.")
                    else:
                        # Load the activation functions
                        for j in range(len(activations[i])):
                            activations[i][j] = L.load_activation(
                                                        activations[i][j], 
                                                        act_params[i][j])
            else:
                architectures = conf["architectures"]
                layers        = conf["layers"].split()
                lay_params    = conf["lay_params"].split()
                nodes         = [int(num) 
                                 for num in conf["nodes"].split()]
                activations   = conf["activations"].split()
                act_params    = conf["act_params"].split()
                # Check for allowed layer types
                if len(layers) - layers.count("dense")     \
                               - layers.count("conv1d")    \
                               - layers.count("maxpool1d") \
                               - layers.count("avgpool1d") \
                               - layers.count("flatten")   \
                               - layers.count("dropout") > 0:
                    raise Exception('Invalid layer type(s) specified. ' \
                                  + 'Allowed options: dense, conv1d,\n' \
                                  + 'maxpool1d, avgpool1d, flatten')
                # Make sure the right number of entries exist
                if len(layers) - layers.count("maxpool1d")            \
                               - layers.count("avgpool1d")            \
                               - layers.count("flatten")              \
                               - layers.count("dropout") != len(nodes):
                    raise Exception("Number of Dense/Conv layers does not " \
                        + "match the number of hidden\nlayers with nodes.")
                if len(layers) != len(lay_params):
                    raise Exception("Number of layer types does not match " \
                        + "the number of layer parameters.")
                else:
                    # Set default layer parameters if needed
                    for j in range(len(layers)):
                        if lay_params[j] == 'None':
                            if layers[j] == 'conv1d':
                                lay_params[j] = 3
                            elif layers[j] == 'maxpool1d' or \
                                 layers[j] == 'avgpool1d':
                                lay_params[j] = 2
                        elif layers[j] == 'dropout':
                            lay_params[j] = float(lay_params[j])
                        else:
                            lay_params[j] = int(lay_params[j])
                if len(activations) != len(nodes):
                    raise Exception("Number of activation functions does "    \
                        + "not match the number of hidden\nlayers with nodes.")
                if len(activations) != len(act_params):
                    raise Exception("Number of activation functions does not " \
                        + "match the number of\nactivation function "          \
                        + "parameters.")
                else:
                    # Load the activation functions
                    for j in range(len(activations)):
                        activations[j] = L.load_activation(activations[j], 
                                                           act_params[j])

            # Learning rate parameters
            lengthscale = conf.getfloat("lengthscale")
            max_lr      = conf.getfloat("max_lr")
            clr_mode    = conf["clr_mode"]
            clr_steps   = conf["clr_steps"]

            # Plotting parameters
            xlabel     = conf["xlabel"]
            if conf["xvals"] in ["None", "none", "False"]:
                fxvals = None
            else:
                # Will be loaded in NN.py
                if os.path.isabs(conf["xvals"]):
                    fxvals = conf["xvals"]
                else:
                    fxvals = inputdir + conf["xvals"]
            ylabel     = conf["ylabel"]
            plot_cases = [int(num) for num in conf["plot_cases"].split()]

            # Generate data set
            if datagen:
                print('\nMode: Generate data\n')
                D.generate_data(inputdir+cfile)

            if processdat:
                print('\nMode: Process data\n')
                D.process_data(inputdir+cfile, datadir, preservedat)

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
                          gridsearch, architectures, 
                          layers, lay_params, activations, act_params, nodes, 
                          lengthscale, max_lr, clr_mode, clr_steps, 
                          epochs, patience, weight_file, resume, 
                          plot_cases, fxvals, xlabel, ylabel, 
                          filters, filt2um)

    return


if __name__ == "__main__":
    #U.limit_mem()
    try:
        MARGE(*sys.argv[1:])
    except MemoryError:
        sys.stderr.write('\nERROR: Memory limit exceeded.')
        sys.exit(1)






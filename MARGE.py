#!/usr/bin/env python3
'''
Driver for MARGE
'''

import sys, os, platform
import time
import configparser
import importlib
import functools
import numpy as np

import tensorflow.keras as keras
K = keras.backend
from tensorflow.keras.regularizers import l1, l2, l1_l2

libdir = os.path.join(os.path.dirname(__file__), 'lib', '')
sys.path.append(libdir)

import loader  as L
import NN
import stats   as S
import utils   as U

sys.path.append(os.path.join(libdir, 'loss'))
import losses


if platform.system() == 'Windows':
    # Windows Ctrl+C fix
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


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
    time_start = time.time()
    # Load configuration file
    defaults = {"L1_regularization" : None, "L2_regularization" : None}
    config = configparser.ConfigParser(defaults, allow_no_value=True)
    config.read_file(open(confile, 'r'))

    # Run everything specified in config file
    for section in config:
        if section != "DEFAULT":
            conf = config[section]
            ### Unpack the variables ###########################################
            verb = conf.getint("verb")
            # Directories
            inputdir  = os.path.join(os.path.abspath(conf["inputdir" ]), '')
            outputdir = os.path.join(os.path.abspath(conf["outputdir"]), '')
            plotdir = os.path.join(U.load_path(conf["plotdir"], outputdir), '')
            datadir = os.path.join(U.load_path(conf["datadir"], outputdir), '')
            preddir = os.path.join(U.load_path(conf["preddir"], outputdir), '')

            ### Create the directories if they do not exist ####################
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

            ### Main options ###################################################
            datagen     = conf.getboolean("datagen")
            cfile       = conf["cfile"]
            processdat  = conf.getboolean("processdat")
            preservedat = conf.getboolean("preservedat")
            NNmodel     = conf.getboolean("NNmodel")
            gridsearch  = conf.getboolean("gridsearch")
            trainflag   = conf.getboolean("trainflag")
            validflag   = conf.getboolean("validflag")
            testflag    = conf.getboolean("testflag")
            optimize    = conf.getint("optimize")
            resume      = conf.getboolean("resume")
            TFRfile     = conf["TFR_file"]
            if TFRfile != '' and TFRfile[-1] != '_':
                TFRfile = TFRfile + '_' # Separator for file names
            buffer_size = conf.getint("buffer")
            ncores      = conf.getint("ncores")
            if  ncores  > os.cpu_count():
                ncores  = os.cpu_count()
            normalize   = conf.getboolean("normalize")
            scale       = conf.getboolean("scale")
            seed        = conf.getint("seed")
            
            ### Files to save ##################################################
            fxmean = U.load_path(conf["fxmean"], inputdir)
            fxstd  = U.load_path(conf["fxstd"],  inputdir)
            fxmin  = U.load_path(conf["fxmin"],  inputdir)
            fxmax  = U.load_path(conf["fxmax"],  inputdir)
            fymean = U.load_path(conf["fymean"], inputdir)
            fystd  = U.load_path(conf["fystd"],  inputdir)
            fymin  = U.load_path(conf["fymin"],  inputdir)
            fymax  = U.load_path(conf["fymax"],  inputdir)
            fsize     = U.load_path(conf["fsize"], inputdir)
            rmse_file = conf["rmse_file"]
            r2_file   = conf["r2_file"]
            statsaxes = conf["statsaxes"]
            if statsaxes not in ['all', 'batch']:
                raise ValueError("statsaxes parameter must be all or batch.")

            ### Data info ######################################################
            ishape = tuple([int(v) for v in conf["ishape"].split(',')])
            oshape = tuple([int(v) for v in conf["oshape"].split(',')])

            if conf["ilog"].lower() in ["true", "t", "false", "f"]:
                ilog = conf.getboolean("ilog")
            elif conf["ilog"].lower() in ["none", ""]:
                ilog = False
            elif conf["ilog"].isdigit():
                ilog = int(conf["ilog"])
            elif any(pun in conf["ilog"] for pun in [",", " ", "\n"]):
                if "," in conf["ilog"]:
                    ilog = [int(num) for num in conf["ilog"].split(',')]
                else:
                    ilog = [int(num) for num in conf["ilog"].split()]
                #if any(num >= inD for num in ilog):
                #    raise ValueError("One or more ilog indices exceed the " + \
                #                     "specified number of inputs.")
            else:
                raise ValueError("ilog specification not understood.")

            if conf["olog"].lower() in ["true", "t", "false", "f"]:
                olog = conf.getboolean("olog")
            elif conf["olog"].lower() in ["none", ""]:
                olog = False
            elif conf["olog"].isdigit():
                olog = int(conf["olog"])
            elif any(pun in conf["olog"] for pun in [",", " ", "\n"]):
                if "," in conf["olog"]:
                    olog = [int(num) for num in conf["olog"].split(',')]
                else:
                    olog = [int(num) for num in conf["olog"].split()]
                #if any(num >= outD for num in olog):
                #    raise ValueError("One or more olog indices exceed the " + \
                #                     "specified number of outputs.")
            else:
                raise ValueError("olog specification not understood.")

            if scale:
                scalelims = [int(num) for num in conf["scalelims"].split(',')]
            else:
                scalelims = [0., 1.] # Won't change the data values
            try:
                filters = conf["filters"].split()
                filtconv = float(conf["filtconv"])
                if verb:
                    print('\nFilters specified; computing performance ' \
                        + 'metrics over integrated bandpasses.\n')
            except:
                filters = None
                filtconv = 1.
                if verb:
                    print('\nFilters not specified; computing performance ' \
                        + 'metrics for each output.\n')

            ### Model info #####################################################
            weight_file = U.load_path(conf["weight_file"], outputdir)
            epochs      = conf.getint("epochs")
            batch_size  = conf.getint("batch_size")
            patience    = conf.getint("patience")

            ### Import the datagen module ######################################
            if datagen or processdat:
                datagenfile = conf["datagenfile"].rsplit(os.sep, 1)
                if len(datagenfile) == 2:
                    pth = U.load_path(datagenfile[0], inputdir)
                    if verb:
                        print("Appending path for module with datagen function:", pth)
                    sys.path.append(pth)
                else:
                    # Look in inputdir first, check lib/datagen/ after
                    sys.path.append(inputdir)
                    sys.path.append(os.path.join(libdir, 'datagen', ''))
                D = importlib.import_module(datagenfile[-1])

            ### Learning rate parameters #######################################
            lengthscale = conf.getfloat("lengthscale")
            max_lr      = conf.getfloat("max_lr")
            clr_mode    = conf["clr_mode"]
            clr_steps   = conf["clr_steps"]

            ### Custom loss functions ##########################################
            model_predict  = None
            model_evaluate = None
            if "lossfunc" in conf.keys():
                # Format: path/to/module.py function_name
                lossfunc = conf["lossfunc"].split()  # [path/to/module.py, function_name]
                if lossfunc[0] == 'mse':
                    lossfunc = keras.losses.MeanSquaredError
                    lossfunc.__name__ = 'mse'
                elif lossfunc[0] == 'mae':
                    lossfunc = keras.losses.MeanAbsoluteError
                    lossfunc.__name__ = 'mae'
                elif lossfunc[0] == 'mape':
                    lossfunc = keras.losses.MeanAbsolutePercentageError
                    lossfunc.__name__ = 'mape'
                elif lossfunc[0] == 'msle':
                    lossfunc = keras.losses.MeanSquaredLogarithmicError
                    lossfunc.__name__ = 'msle'
                elif lossfunc[0] in ['maxmse', 'm3se', 'mslse', 'mse_per_ax', 'maxse']:
                    lossname = lossfunc[0]
                    lossfunc = getattr(losses, lossfunc[0])
                    lossfunc.__name__ = lossname
                elif lossfunc[0] in ['heteroscedastic', 'heteroscedastic_loss']:
                    lossfunc = functools.partial(losses.heteroscedastic_loss, D=np.product(oshape), N=batch_size)
                    lossfunc.__name__ = 'heteroscedastic_loss'
                else:
                    if lossfunc[0][-3:] == '.py':
                        lossfunc[0] = lossfunc[0][:-3]   # path/to/module
                    mod = lossfunc[0].rsplit(os.sep, 1) # [path/to, module]
                    if len(mod) == 2:
                        pth = U.load_path(mod[0], inputdir)
                        if verb:
                            print("Appending path for module with loss function:", pth)
                        sys.path.append(pth)
                    mod = importlib.import_module(mod[-1])
                    if len(lossfunc) == 2:
                        lossname = lossfunc[-1]
                        lossfunc = getattr(mod, lossfunc[-1])
                        lossfunc.__name__ = lossname
                    else:
                        lossfunc = getattr(mod, 'loss')
                        lossfunc.__name__ = 'loss'
            else:
                lossfunc = None

            ### Grid search parameters #########################################
            if gridsearch:
                if optimize:
                    raise ValueError("Cannot use both grid search and Bayesian hyperparameter optimization.")
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
                # Check that the parameters are valid
                lay_params, activations = U.prepare_gridsearch(architectures,
                                                               layers,
                                                               lay_params,
                                                               nodes,
                                                               activations,
                                                               act_params)
            else:
                ### Single neural network's parameters #########################
                architectures = conf["architectures"]
                layers        = conf["layers"].split()
                lay_params    = conf["lay_params"].split()
                nodes         = [int(num)
                                 for num in conf["nodes"].split()]
                activations   = conf["activations"].split()
                act_params    = conf["act_params"].split()
                # Check that the parameters are valid
                lay_params, activations = U.prepare_layers(layers, lay_params,
                                                           nodes, activations,
                                                           act_params)
            ### Regularization parameters ######################################
            L1_regularization = conf["L1_regularization"]
            L2_regularization = conf["L2_regularization"]
            
            if L1_regularization is not None:
                if L1_regularization.lower() not in ["none", "false", "f", ""]:
                    if L1_regularization.lower() in ["true", "t"]:
                        # Default value
                        L1_regularization = 0.01
                    else:
                        # User-specified value
                        L1_regularization = float(L1_regularization)
                        # If it's 0, don't waste the CPU cycles
                        if L1_regularization == 0:
                            L1_regularization = None
            
            if L2_regularization is not None:
                if L2_regularization.lower() not in ["none", "false", "f", ""]:
                    if L2_regularization.lower() in ["true", "t"]:
                        L2_regularization = 0.01
                    else:
                        L2_regularization = float(L2_regularization)
                        if L2_regularization == 0:
                            L2_regularization = None
            
            if isinstance(L1_regularization, float) and isinstance(L2_regularization, float):
                kernel_regularizer = l1_l2(l1=L1_regularization, l2=L2_regularization)
            elif isinstance(L1_regularization, float):
                kernel_regularizer = l1(l=L1_regularization)
            elif isinstance(L2_regularization, float):
                kernel_regularizer = l2(l=L2_regularization)
            else:
                kernel_regularizer = None
            
            ### Bayesian optimization parameters ###############################
            if optimize:
                if gridsearch:
                    raise ValueError("Cannot use both grid search and Bayesian hyperparameter optimization.")
                if "optfunc" in conf.keys():
                    # Format: path/to/module.py function_name
                    optfunc = conf["optfunc"].split()  # [path/to/module.py, function_name]
                    if optfunc[0][-3:] == '.py':
                        optfunc[0] = optfunc[0][:-3]   # path/to/module
                    mod = optfunc[0].rsplit(os.sep, 1) # [path/to, module]
                    if len(mod) == 2:
                        pth = U.load_path(mod[0], inputdir)
                        if verb:
                            print("Appending path for module with optimization function:", pth)
                        sys.path.append(pth)
                    mod = importlib.import_module(mod[-1])
                    if len(optfunc) == 2:
                        optfunc = getattr(mod, optfunc[-1])
                    else:
                        optfunc = getattr(mod, 'objective')
                else:
                    optfunc = None
                optngpus = conf.getint("optngpus")
                optnlays = [int(val) for val in conf["optnlays"].split()]
                if "optlayer" in conf.keys():
                    if conf['optlayer'] not in [None, 'none', 'None']:
                        optlayer = conf['optlayer'].split()
                    else:
                        optlayer = None
                else:
                    optlayer = None
                if "optnnode" in conf.keys():
                    if conf["optnnode"] not in [None, 'none', 'None']:
                        optnnode = [int(val) for val in conf["optnnode"].split()]
                    else:
                        optnnode = None
                else:
                    optnnode = None
                optactiv = conf["optactiv"].split()
                try:
                    optminlr = conf.getfloat("optminlr")
                except:
                    optminlr = None
                try:
                    optmaxlr = conf.getfloat("optmaxlr")
                except:
                    optmaxlr = None
                try:
                    optactrng = [float(val) for val in conf["optactrng"].split()]
                except:
                    optactrng = None
                try:
                    opttime = conf.getint("opttime")
                except:
                    opttime = None
                try:
                    optmaxconvnode = conf.getint("optmaxconvnode")
                except:
                    optmaxconvnode = None
            else:
                optfunc  = None
                optngpus = None
                optnlays = None
                optlayer = None
                optnnode = None
                opttime  = None
                optmaxconvnode = None
                optactiv = None
                optminlr = None
                optmaxlr = None
                optactrng = None

            ### Plotting parameters ############################################
            xlabel     = conf["xlabel"]
            if conf["xvals"].lower() in ["none", "false", "f", ""]:
                fxvals = None
            else:
                # Will be loaded in NN.py
                fxvals = U.load_path(conf["xvals"], inputdir)
            ylabel     = conf["ylabel"]
            if conf["plot_cases"].lower() in ["none", "false", "f", ""]:
                plot_cases = None
            else:
                plot_cases = [int(num) for num in conf["plot_cases"].split()]
            if conf["smoothing"].lower() in ["none", "false", "f", ""]:
                smoothing = False
            else:
                smoothing = conf.getint("smoothing")

            ### Generate data set ##############################################
            if datagen:
                print('\nMode: Generate data\n')
                D.generate_data(inputdir+cfile)

            if processdat:
                print('\nMode: Process data\n')
                D.process_data(inputdir+cfile, datadir, preservedat)
            
            ### Get data set sizes #############################################
            set_nbatches = U.get_data_set_sizes(fsize, datadir, ishape, oshape, 
                                                batch_size, ncores)
            train_batches, valid_batches, test_batches = set_nbatches
            
            ### Ensure TFRecords exist, and that they don't need updating ######
            fTFR = U.check_TFRecords(inputdir, TFRfile, datadir, ilog, olog, 
                                     batch_size, train_batches, valid_batches, test_batches)

            ### Train model(s) #################################################
            if NNmodel:
                print('\nMode: Neural network model\n')
                nn = NN.driver(inputdir, outputdir, datadir, plotdir, preddir,
                          trainflag, validflag, testflag,
                          normalize, fxmean, fxstd, fymean, fystd,
                          scale, fxmin, fxmax, fymin, fymax, scalelims,
                          rmse_file, r2_file, statsaxes,
                          ishape, oshape, ilog, olog,
                          fTFR, batch_size, set_nbatches, ncores, buffer_size,
                          optimize, optfunc, optngpus, opttime, optnlays, optlayer,
                          optnnode, optactiv, optactrng, optminlr, optmaxlr, optmaxconvnode,
                          gridsearch, architectures,
                          layers, lay_params, activations, nodes, kernel_regularizer, 
                          lossfunc, lengthscale, max_lr, clr_mode, clr_steps,
                          model_predict, model_evaluate, 
                          epochs, patience, weight_file, resume,
                          plot_cases, fxvals, xlabel, ylabel, smoothing,
                          filters, filtconv, verb)

    time_end = time.time()
    time_total = time_end - time_start
    print("Total time elapsed:", int(time_total//3600), "hrs", int(time_total//60%60), "min", str(time_total%60)[:6], "sec")

    return nn


if __name__ == "__main__":
    MARGE(*sys.argv[1:])

#!/usr/bin/env python3
'''
Driver for MARGE
'''

import sys, os, platform
import datetime
import time
import configparser
import importlib
import functools
import logging
import types
import numpy as np

import tensorflow as tf
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

import custom_logger as CL
logging.setLoggerClass(CL.MARGE_Logger)


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
    # Load configuration file - set reasonable defaults
    defaults = {"datadir"   : "data"  , "datagen"     : False,
                "preddir"   : "pred"  , "processdat"  : False,
                "plotdir"   : "plots" , "preservedat" : True , 
                "resume"    : False   , "cfile"       : ""   , 
                "NNmodel"   : False   , "gridsearch"  : False, 
                "TFR_file"  : ""      , "optimize"    : 0    , 
                "normalize" : False   , "ilog"        : None , 
                "scale"     : False   , "olog"        : None , 
                "scalelims" : [0., 1.], "buffer"      : 15   , 
                "trainflag" : False   , "ncores"      : 1    , 
                "validflag" : False   , "L1_regularization" : None, 
                "testflag"  : False   , "L2_regularization" : None, 
                "seed"      : np.random.randint(low=0, high=9999999), 
                "fxmean" : 'xmean.npy'  , "fxstd"     : 'xstd.npy',
                "fxmin"  : 'xmin.npy'   , "fxmax"     : 'xmax.npy', 
                "fymean" : 'ymean.npy'  , "fystd"     : 'ystd.npy',
                "fymin"  : 'ymin.npy'   , "fymax"     : 'ymax.npy', 
                "fsize"  : "datsize.npy", "statsaxes" : "all" , 
                "rmse_file"   : "rmse"  , "r2_file"   : "r2", 
                "weight_file" : "nn_weights.keras", 
                "plot_cases"  : None , "smoothing" : 0, 
                "use_cpu"     : False, "verb"      : 2}
    config = configparser.ConfigParser(defaults, allow_no_value=True, 
                                       inline_comment_prefixes=('#', ';'))
    config.read_file(open(confile, 'r'))

    # Run everything specified in config file
    nsec_proc = 0 # Counter for processed sections, to ensure we only set up 1 logger
    for section in config:
        if section != "DEFAULT":
            conf = config[section]
            ### Unpack the variables ###########################################
            verb = conf.getint("verb")
            
            ### Main directories################################################
            inputdir  = os.path.join(os.path.abspath(conf["inputdir" ]), '')
            outputdir = os.path.join(os.path.abspath(conf["outputdir"]), '')
            
            #U.make_dir(outputdir)
            os.makedirs(outputdir, exist_ok=True)

            ### Set up logger ##################################################
            if not nsec_proc:
                flog = os.path.join(outputdir, 
                                    datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.log')
                
                # Set logging level
                if verb >= 5:
                    # Print everything that might be useful
                    # In the future, may add different logging levels
                    log_level = logging.DEBUG
                elif verb >= 2:
                    # 2: print the important details
                    # 3: print a little bit more info
                    # 4: prints the defaults used if a parameter isn't provided
                    log_level = logging.INFO
                elif verb:
                    # 1: Minimal printing
                    log_level = logging.WARNING
                else:
                    # 0: Only print essential messages
                    log_level = logging.ERROR
                # Set logging format
                log_fmt = '[%(asctime)s] [%(name)s.%(funcName)s] [%(levelname)s] %(message)s'
                date_fmt = '%Y/%m/%d %H:%M:%S'
                # Set up logging handlers
                file_handler   = logging.FileHandler(flog)
                stream_handler = logging.StreamHandler(sys.stdout)
                file_handler  .setLevel(log_level)
                stream_handler.setLevel(log_level)
                file_handler  .setFormatter(logging.Formatter(log_fmt, date_fmt))
                stream_handler.setFormatter(logging.Formatter(log_fmt, date_fmt))
                normal_handlers = [file_handler, stream_handler]
                # Set up blank-line handlers
                blank_file_handler   = logging.FileHandler(flog)
                blank_stream_handler = logging.StreamHandler(sys.stdout)
                blank_file_handler  .setLevel(logging.DEBUG)
                blank_stream_handler.setLevel(logging.DEBUG)
                blank_file_handler  .setFormatter(logging.Formatter(fmt=''))
                blank_stream_handler.setFormatter(logging.Formatter(fmt=''))
                blank_handlers = [blank_file_handler, blank_stream_handler]

                logger = logging.getLogger('MARGE')
                for handler in normal_handlers:
                    logger.addHandler(handler)
                logger.setLevel(log_level)
                logger.normal_handlers = normal_handlers
                logger.blank_handlers  = blank_handlers
                logger.info("Beginning MARGE\n")

                if verb >= 4:
                    defaults_str = "\n".join(["    " + key + " = " + str(defaults[key]) for key in list(defaults.keys())])
                    logger.info("Using the following defaults for any parameters " + \
                                "that are not specified:\n" + defaults_str + "\n")
            
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
                logger.warning("Requested " + str(ncores) + " cores, but found only " + \
                            str(os.cpu_count()) + " available.  Reducing accordingly.")
                ncores  = os.cpu_count()
            normalize   = conf.getboolean("normalize")
            scale       = conf.getboolean("scale")
            seed        = conf.getint("seed")

            # Sanity check: if no GPUs, ensure the user has enabled use_cpu
            ngpu_detected = len(tf.config.list_physical_devices('GPU'))
            use_cpu = conf.getboolean("use_cpu")
            if not use_cpu and not ngpu_detected:
                logger.error("No GPUs detected by TensorFlow, but `use_cpu` " + \
                             "is not set to True in the configuration file. " + \
                             "If you want to use the CPU, set that parameter" + \
                             " and re-run MARGE.  If you intended to use a "  + \
                             "GPU, your system and/or environment may not be" + \
                             " set up correctly.")
                sys.exit(1)
            if use_cpu and ngpu_detected:
                logger.warning("GPU detected, but configuration file says to use the CPU.  " +\
                               "MARGE execution will be slower compared to using the GPU.\n")
            if use_cpu and trainflag:
                logger.warning("Configuration file says to use the CPU.  " +\
                               "NN training may take a very long time, " +\
                               "depending on the problem.\n")

            ### Sub-directories ################################################
            plotdir = os.path.join(U.load_path(conf["plotdir"], outputdir), '')
            datadir = os.path.join(U.load_path(conf["datadir"], outputdir), '')
            preddir = os.path.join(U.load_path(conf["preddir"], outputdir), '')

            ### Create the directories if they do not exist ####################
            U.make_dir(inputdir)
            U.make_dir(os.path.join(inputdir, 'TFRecords', ''))
            U.make_dir(plotdir)
            U.make_dir(datadir)
            U.make_dir(os.path.join(datadir, 'train', ''))
            U.make_dir(os.path.join(datadir, 'valid', ''))
            U.make_dir(os.path.join(datadir, 'test' , ''))
            U.make_dir(preddir)
            U.make_dir(os.path.join(preddir, 'valid', ''))
            U.make_dir(os.path.join(preddir, 'test' , ''))
            
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
                logger.error("statsaxes parameter must be all or batch.\nReceived: " + statsaxes)
                sys.exit(1)

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
            else:
                logger.error("ilog specification not understood.\n" +\
                    "Expected something like True, False, None, or 0 2 3 5\n" +\
                    "Received: " + str(ilog))
                sys.exit(1)

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
            else:
                logger.error("olog specification not understood.\n" +\
                    "Expected something like True, False, None, or 0 2 3 5\n" +\
                    "Received: " + str(olog))
                sys.exit(1)

            if scale:
                try:
                    scalelims = [float(num) for num in conf["scalelims"].split(',')]
                    assert len(scalelims) == 2, "Expected 2 values for " +\
                             "`scalelims` but received " + str(len(scalelims))
                except Exception as e:
                    logger.critical("Format of `scalelims` cannot be understood.\n" +\
                                    "Expected: 2 comma-separated floats.\n" +\
                                    "Received: " + conf["scalelims"] + "\n" +\
                                    "Error: " + e)
                    sys.exit(1)
            else:
                scalelims = [0., 1.] # Won't change the data values
            
            try:
                filters = conf["filters"].split()
                filtconv = float(conf["filtconv"])
                logger.info('Filters specified; computing performance ' \
                          + 'metrics over integrated bandpasses.')
            except:
                filters = None
                filtconv = 1.
                logger.info('Filters not specified; computing performance ' \
                          + 'metrics for each output.')

            ### Model info #####################################################
            weight_file = U.load_path(conf["weight_file"], outputdir)
            # Update this check if ONNX -> Keras ever becomes supported in the future
            if weight_file[-6:] != '.keras':
                logger.critical('Weight file must end in .keras.\nReceived: '+weight_file)
                sys.exit(1)
            epochs      = conf.getint("epochs")
            batch_size  = conf.getint("batch_size")
            patience    = conf.getint("patience")

            ### Import the datagen module ######################################
            if datagen or processdat:
                datagenfile = conf["datagenfile"].rsplit(os.sep, 1)
                if len(datagenfile) == 2:
                    pth = U.load_path(datagenfile[0], inputdir)
                    if verb >= 3:
                        logger.info("Appending path for module with datagen function:", pth)
                    sys.path.append(pth)
                else:
                    # Look in inputdir first, check lib/datagen/ after
                    sys.path.append(inputdir)
                    sys.path.append(os.path.join(libdir, 'datagen', ''))
                D = importlib.import_module(datagenfile[-1])

            ### Learning rate parameters #######################################
            try:
                min_lr = conf.getfloat("lengthscale")
            except:
                min_lr = conf.getfloat("min_lr")
            try:
                max_lr = conf.getfloat("max_lr")
            except:
                max_lr = min_lr
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
                        if verb >= 3:
                            logger.info("Appending path for module with loss function:", pth)
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
                    logger.critical("Cannot use both grid search and Bayesian hyperparameter optimization.")
                    sys.exit(1)
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
                lay_params, activations = U.prepare_gridsearch(layers,
                                                               lay_params,
                                                               nodes,
                                                               activations,
                                                               act_params)
            else:
                ### Single neural network's parameters #########################
                # Sanity check for multiple models specified
                if any('\n' in var for var in [conf["layers"]       [:-1], 
                                               conf["lay_params"]   [:-1], 
                                               conf["nodes"]        [:-1], 
                                               conf["act_params"]   [:-1]]):
                    logger.error("Grid search is set to False, but new "   +\
                                 "lines were found in one or more config " +\
                                 "parameters for the model.  Ensure that " +\
                                 "`layers`, `lay_params`, "+\
                                 "`nodes`, and `act_params` each have all "+\
                                 "parameters on a single line.")
                    sys.exit(1)
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
                    logger.critical("Cannot use both grid search and Bayesian hyperparameter optimization.")
                    sys.exit(1)
                if "optfunc" in conf.keys():
                    # Format: path/to/module.py function_name
                    optfunc = conf["optfunc"].split()  # [path/to/module.py, function_name]
                    if optfunc[0][-3:] == '.py':
                        optfunc[0] = optfunc[0][:-3]   # path/to/module
                    mod = optfunc[0].rsplit(os.sep, 1) # [path/to, module]
                    if len(mod) == 2:
                        pth = U.load_path(mod[0], inputdir)
                        if verb >= 3:
                            logger.info("Appending path for module with optimization function:", pth)
                        sys.path.append(pth)
                    mod = importlib.import_module(mod[-1])
                    if len(optfunc) == 2:
                        optfunc = getattr(mod, optfunc[-1])
                    else:
                        optfunc = getattr(mod, 'objective')
                else:
                    optfunc = None
                optngpus = conf.getint("optngpus")
                # Check that there are that many GPUs available
                if ngpu_detected < optngpus:
                    logger.error("Requested to use " + str(optngpus) + " for " +\
                                 "the Bayesian optimization, but only " + \
                                 str(ngpu_detected) + " GPUs detected.")
                    sys.exit(1)
                optnlays = [int(val) for val in conf["optnlays"].split()]
                assert len(optnlays) in [1, 2], "Expected to receive 1 or 2 " +\
                            "ints for `optnlays`.\nReceived: " + str(optnlays)
                if "optlayer" in conf.keys():
                    if conf['optlayer'] not in [None, 'none', 'None']:
                        optlayer = conf['optlayer'].split()
                    else:
                        optlayer = None
                    if len(optlayer) > optnlays[-1]:
                        logger.warning("Specified " + str(len(optlayer)) + \
                                       " layers to use during optimization, " +\
                                       "but the maximum number of layers to " +\
                                       "consider during the optimization is " +\
                                       "specified to be " + str(optnlays[-1]) +\
                                       ".  The last " + str(optnlays[-1] - len(optlayer)) +\
                                       " specified layers will not be considered.")
                    elif len(optlayer) < optnlays[-1]:
                        logger.error("Specified Bayesian optimization with up to " + \
                                     str(optnlays[-1]) + " layers, but only " + \
                                     str(len(optlayer)) + " layers specified for `optlayer`.")
                        sys.exit(1)
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
            xlabel = conf["xlabel"]
            if conf["xvals"].lower() in ["none", "false", "f", ""]:
                fxvals = None
            else:
                # Will be loaded in NN.py
                fxvals = U.load_path(conf["xvals"], inputdir)
            ylabel = conf["ylabel"]
            if conf["plot_cases"].lower() in ["none", "false", "f", ""]:
                plot_cases = None
            else:
                plot_cases = [int(num) for num in conf["plot_cases"].split()]
            if conf["smoothing"].lower() in ["none", "false", "f", ""]:
                smoothing = False
            else:
                smoothing = conf.getint("smoothing")

            time_config = time.time()
            time_elapsed = time_config - time_start
            logger.newline()
            logger.info("Time spent reading configuration file: "+ \
                        str(np.round(time_elapsed, 6)) + " sec")
            ### Generate data set ##############################################
            if datagen:
                logger.info('Mode: Generate data\n')
                D.generate_data(inputdir+cfile)
            # Outide of if block, to allow each mode to be independently timed
            time_datagen = time.time()
            if datagen:
                time_elapsed = time_datagen - time_config
                logger.newline()
                logger.info("Time spent generating data: "+ \
                            str(int(time_elapsed//3600)) + " hrs " + \
                            str(int(time_elapsed//60%60)) + " min " + \
                            str(time_elapsed%60)[:6] + " sec")

            if processdat:
                logger.info('Mode: Process data\n')
                D.process_data(inputdir+cfile, datadir, preservedat)
            # Outide of if block, to allow each mode to be independently timed
            time_processdat = time.time()
            if processdat:
                time_elapsed = time_processdat - time_datagen
                logger.newline()
                logger.info("Time spent processing data: "+ \
                            str(int(time_elapsed//3600)) + " hrs " + \
                            str(int(time_elapsed//60%60)) + " min " + \
                            str(time_elapsed%60)[:6] + " sec")
            
            ### Get data set sizes #############################################
            set_nbatches = U.get_data_set_sizes(fsize, datadir, ishape, oshape, 
                                                batch_size, ncores)
            train_batches, valid_batches, test_batches = set_nbatches
            
            time_batches = time.time()
            time_elapsed = time_batches - time_processdat
            logger.newline()
            logger.info("Time spent getting data set sizes: "+ \
                        str(int(time_elapsed//3600)) + " hrs " + \
                        str(int(time_elapsed//60%60)) + " min " + \
                        str(time_elapsed%60)[:6] + " sec")
            logger.newline()

            ### Ensure TFRecords exist, and that they don't need updating ######
            fTFR = U.check_TFRecords(inputdir, TFRfile, datadir, ilog, olog, 
                                     batch_size, train_batches, valid_batches, test_batches)
            
            time_tfr = time.time()
            time_elapsed = time_tfr - time_batches
            logger.newline()
            logger.info("Time spent checking TFRecords files: "+ \
                        str(int(time_elapsed//3600)) + " hrs " + \
                        str(int(time_elapsed//60%60)) + " min " + \
                        str(time_elapsed%60)[:6] + " sec")

            ### Train model(s) #################################################
            if NNmodel:
                logger.newline()
                logger.info('Mode: Neural network model\n')
                nn = NN.driver(inputdir, outputdir, datadir, plotdir, preddir,
                          trainflag, validflag, testflag,
                          normalize, fxmean, fxstd, fymean, fystd,
                          scale, fxmin, fxmax, fymin, fymax, scalelims,
                          rmse_file, r2_file, statsaxes,
                          ishape, oshape, ilog, olog,
                          fTFR, batch_size, set_nbatches, ncores, buffer_size,
                          optimize, optfunc, optngpus, opttime, optnlays, optlayer,
                          optnnode, optactiv, optactrng, optminlr, optmaxlr, optmaxconvnode,
                          gridsearch, 
                          layers, lay_params, activations, nodes, kernel_regularizer, 
                          lossfunc, min_lr, max_lr, clr_mode, clr_steps,
                          model_predict, model_evaluate, 
                          epochs, patience, weight_file, resume,
                          plot_cases, fxvals, xlabel, ylabel, smoothing,
                          filters, filtconv, use_cpu, verb)
                time_nn = time.time()
                time_elapsed = time_nn - time_tfr
                logger.newline()
                logger.info("Time spent on NN training, validation, and/or testing: "+ \
                            str(int(time_elapsed//3600)) + " hrs " + \
                            str(int(time_elapsed//60%60)) + " min " + \
                            str(time_elapsed%60)[:6] + " sec")
            nsec_proc += 1

    time_end = time.time()
    time_total = time_end - time_start
    logger.newline()
    logger.info("Total time elapsed: " + str(int(time_total//3600)) + " hrs " + \
                str(int(time_total//60%60)) + " min " + str(time_total%60)[:6] + " sec")

    return nn


if __name__ == "__main__":
    MARGE(*sys.argv[1:])

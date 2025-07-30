"""
This file contains 3 callbacks:

CyclicLR
comes from https://github.com/bckenstler/CLR
It implements a Cyclical Learning Rate (CLR) to improve model training.
It has been modified to be pickle-able (see ModelCheckpointEnhanced).

SignalStopping
comes from https://github.com/keras-team/keras/issues/51#issuecomment-430041670
It implements training stopping via Ctrl+C or a specified stop file.

ModelCheckpointEnhanced
based on https://gist.github.com/eyalzk/215a53225b38c411f3cc7e29de1f8083
It implements a model checkpoint system that stores information about 
other callbacks to enable the resumption of training.

"""

import sys, os
import time
import logging
logging.setLoggerClass(logging.Logger)
import signal
import numpy as np
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K
import pickle
import glob

import custom_logger as CL
logging.setLoggerClass(CL.MARGE_Logger)

logger = logging.getLogger('MARGE.'+__name__)


class CyclicLR(Callback):
    """
    From: https://github.com/bckenstler/CLR
    It has been modified to be pickle-able.

    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., 
                 mode='triangular', gamma=1., scale_fn=None, 
                 scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr   = base_lr
        self.max_lr    = max_lr
        self.step_size = step_size
        self.mode      = mode
        self.gamma     = gamma
        if scale_fn is None:
            if   self.mode == 'triangular':
                #self.scale_fn   = lambda x: 1.
                self.scale_fn   = self.triangular
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                #self.scale_fn   = lambda x: 1/(2.**(x-1))
                self.scale_fn   = self.triangular2
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                #self.scale_fn   = lambda x: gamma**(x)
                self.scale_fn   = self.exp_range
                self.scale_mode = 'iterations'
            else:
                raise ValueError("Unknown CLR mode provided.\nGiven: "+self.mode+\
                                 "\nOptions: triangular, triangular2, exp_range")
        else:
            self.scale_fn   = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history        = {}

        self._reset()

    # pickle-able alternative to lambda functions
    def triangular(self, x):
        return 1.
    def triangular2(self, x):
        return 1. / (2.**(x-1))
    def exp_range(self, x):
        return self.gamma**(x)

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2*self.step_size))
        x     = np.abs(self.clr_iterations / self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                   np.maximum(0, (1-x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                   np.maximum(0, (1-x)) * self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            self.model.optimizer.learning_rate.assign(np.asarray(self.base_lr, dtype=self.model.optimizer.learning_rate.dtype))
        else:
            self.model.optimizer.learning_rate.assign(np.asarray(self.clr(), dtype=self.model.optimizer.learning_rate.dtype))
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(
                                          K.get_value(self.model.optimizer.learning_rate))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        self.model.optimizer.learning_rate.assign(np.asarray(self.clr(), dtype=self.model.optimizer.learning_rate.dtype))


class SignalStopping(Callback):
	'''Stop training when an interrupt signal (or other) was received
		# Arguments
		sig: the signal to listen to. Defaults to signal.SIGINT.
		doubleSignalExits: Receiving the signal twice exits the python
			process instead of waiting for this epoch to finish.
		patience: number of epochs with no improvement
			after which training will be stopped.
		verbose: verbosity mode.
	'''
	# From swilson314 on Github:
    # https://github.com/keras-team/keras/issues/51#issuecomment-430041670
	def __init__(self, sig=signal.SIGINT, doubleSignalExits=False, verbose=0, 
                       stop_file=None,    stop_file_delta=10):
		super(SignalStopping, self).__init__()
		self.signal_received = False
		self.verbose = verbose
		self.doubleSignalExits = doubleSignalExits
		self.stop_file = stop_file
		self.stop_file_time = time.time()
		self.stop_file_delta = stop_file_delta
		def signal_handler(sig, frame):
			if self.signal_received and self.doubleSignalExits:
				logger.warning('\nReceived signal to stop ' + str(sig) + \
                            ' twice. Exiting..')
				exit(sig)
			self.signal_received = True
			logger.warning('\nReceived signal to stop: ' + str(sig))
		signal.signal(signal.SIGINT, signal_handler)
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if self.stop_file is not None:
			# Checking file system is slow in training loop
            # Only check after certain time has elapsed
			delta = time.time() - self.stop_file_time
			if delta>self.stop_file_delta:
				self.stop_file_time += delta
				if os.path.isfile(self.stop_file):
					self.signal_received = True
		if self.signal_received:
			self.stopped_epoch = epoch
			self.model.stop_training = True

	def on_train_end(self, logs={}):
		if self.stopped_epoch > 0:
			logger.warning('Epoch %05d: stopping due to signal' % (self.stopped_epoch))


class ModelCheckpointEnhanced(ModelCheckpoint):
    # Based on code from eyalzk on Github:
    # https://gist.github.com/eyalzk/215a53225b38c411f3cc7e29de1f8083
    def __init__(self, *args, **kwargs):
        # Added arguments
        self.callbacks_to_save  = kwargs.pop('callbacks_to_save')
        self.callbacks_filepath = kwargs.pop('callbacks_filepath')
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        # Run normal flow:
        super().on_epoch_end(epoch,logs)

        # If a checkpoint was saved, save also the callback(s)
        for i in range(len(self.callbacks_to_save)):
            filepath = self.callbacks_filepath[i].format(epoch=epoch+1, **logs)
            if self.epochs_since_last_save == 0 and epoch!=0:
                if self.save_best_only:
                    # Save current point
                    current = logs.get(self.monitor)
                    if current == self.best:
                        # Delete old save points!
                        for foo in glob.glob(self.callbacks_filepath[i].
                                             rsplit('.', 2)[0] + '.epoch*.pickle'):
                            os.remove(foo)
                        # Note, there might be some cases where the last 
                        # statement will save on unwanted epochs.
                        # However, in the usual case where your monitoring 
                        # value space is continuous this is not likely
                        with open(filepath, "wb") as f:
                            pickle.dump(self.callbacks_to_save[i], f)
                else:
                    with open(filepath, "wb") as f:
                        pickle.dump(self.callbacks_to_save[i], f)



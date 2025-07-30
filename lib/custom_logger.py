import sys, os
import copy
import logging
import types


class MARGE_Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET, 
                 fmt='[%(asctime)s] [%(name)s.%(funcName)s] [%(levelname)s] %(message)s', 
                 date_fmt='%Y/%m/%d %H:%M:%S'):
        super().__init__(name, level)
        self.ready = False
        self.fmt = fmt
        self.date_fmt = date_fmt
        self.blank_handlers  = []
        
    def setup(self):
        # Set up handlers for blank new lines
        if not len(self.blank_handlers):
            if not len(self.parent.handlers):
                # No handlers - it isn't a MARGE logger, so ignore
                return
            for handler in self.parent.handlers:
                if 'baseFilename' in handler.__dict__.keys():
                    self.flog = handler.__dict__['baseFilename']
                    self.level = handler.__dict__['level']
            try:
                _ = self.flog
            except:
                return
            # Set up blank-line handlers
            blank_file_handler   = logging.FileHandler(self.flog)
            blank_stream_handler = logging.StreamHandler(sys.stdout)
            blank_file_handler  .setLevel(logging.DEBUG)
            blank_stream_handler.setLevel(logging.DEBUG)
            blank_file_handler  .setFormatter(logging.Formatter(fmt=''))
            blank_stream_handler.setFormatter(logging.Formatter(fmt=''))
            self.blank_handlers = [blank_file_handler, blank_stream_handler]
        # Now ready to use this logger
        self.ready = True
    
    def debug(self, msg, *args, **kwargs):
        if not self.ready:
            self.setup()
        super().debug(msg, stacklevel=2, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        if not self.ready:
            self.setup()
        super().info(msg, stacklevel=2, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        if not self.ready:
            self.setup()
        super().warning(msg, stacklevel=2, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        if not self.ready:
            self.setup()
        super().error(msg, stacklevel=2, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        if not self.ready:
            self.setup()
        super().critical(msg, stacklevel=2, *args, **kwargs)
    
    def newline(self, level='info'):
        """
        Inserts a blank line into the log.  Used for readability.
        
        Inputs
        ------
        level: string.  Logging level.  
                        Options: debug, info, warning, error, critical
        """
        if not self.ready:
            self.setup()
            self.ready = True
        assert level.lower() in ['debug', 'info', 'warning', 'error', 'critical']
        current_handlers = copy.copy(self.handlers)
        if len(current_handlers):
            for handler in current_handlers:
                self.removeHandler(handler)
        # Switch to blank handlers
        self.propagate = False
        for handler in self.blank_handlers:
            self.addHandler(handler)
        
        # Output blank line at the proper logging level
        method = getattr(self, level.lower())
        method('')

        # Switch back
        for handler in self.blank_handlers:
            self.removeHandler(handler)
        if len(current_handlers):
            for handler in current_handlers:
                self.addHandler(handler)
        self.propagate = True


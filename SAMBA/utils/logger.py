# -*- coding: utf-8 -*-
"""
Logging utilities
"""

import os
import logging


def get_logger(root, name=None, debug=True):
    """
    Create a logger for training and evaluation
    
    Args:
        root: Root directory for log files
        name: Logger name
        debug: If True, show DEBUG and INFO in screen; if False, show DEBUG in file and info in both screen&file
    
    Returns:
        logger: Configured logger instance
    """
    # Create a logger
    logger = logging.getLogger(name)
    # critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)
    
    # Define the format
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    
    # Create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # Create a handler for write log to file
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    
    console_handler.setFormatter(formatter)
    
    # Add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    
    return logger

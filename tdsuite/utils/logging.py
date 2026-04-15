"""Logging utilities for technical debt classification."""

import logging
import os
import sys
from datetime import datetime


def setup_logging(output_dir=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log files (optional)
        level: Logging level (default: INFO)
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger("tdsuite")
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"tdsuite_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 
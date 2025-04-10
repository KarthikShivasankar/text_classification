#!/usr/bin/env python
"""
Script to run training with the tdsuite package.
"""

import os
import sys
from tdsuite.train import main

if __name__ == "__main__":
    # Set command line arguments
    sys.argv = [
        "tdsuite-train",
        "--data_file", "karths/binary-10IQR-TD",
        "--model_name", "distilbert-base-uncased",
        "--classification_type", "binary",
        "--numeric_labels",
        "--output_dir", "outputs/binary"
    ]
    
    # Run the training
    main() 
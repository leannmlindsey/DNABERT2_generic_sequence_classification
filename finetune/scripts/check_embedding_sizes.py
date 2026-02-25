#!/usr/bin/env python3
"""
Find all .npy and .npz files under a given directory and print their shapes.
This helps identify embedding dimensions used for linear probe models.

Usage:
    python check_embedding_sizes.py /path/to/search
"""

import sys
import os
import numpy as np

def check_file(filepath):
    if filepath.endswith('.npy'):
        arr = np.load(filepath)
        print(f"{filepath}\n  shape: {arr.shape}\n")
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        for key in data.files:
            print(f"{filepath} [{key}]\n  shape: {data[key].shape}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_embedding_sizes.py /path/to/search")
        sys.exit(1)

    search_dir = sys.argv[1]

    for root, dirs, files in os.walk(search_dir):
        for f in sorted(files):
            if f.endswith('.npy') or f.endswith('.npz'):
                filepath = os.path.join(root, f)
                try:
                    check_file(filepath)
                except Exception as e:
                    print(f"{filepath}\n  ERROR: {e}\n")

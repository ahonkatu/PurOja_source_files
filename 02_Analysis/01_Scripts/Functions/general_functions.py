#Not optimazed original code
#Compared to this, ChatGPT gave very short code for optimization
#This was used in making the firts picles, so this stayes as it is
# General arr lib
import numpy as np

# Compile py code
from numba import jit, njit
from numba import prange

# Multithreading
import dask.array as da
import dask as dk
dk.config.set(scheduler='processes')

from dask_image.ndfilters import generic_filter as d_gf

from collections import deque

from skimage.filters import gabor
from skimage.restoration import  denoise_bilateral

import datetime

from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import generic_filter
import scipy.stats.mstats as ms
from scipy.stats import skew
import scipy.ndimage.morphology as morph
from scipy import ndimage
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import random
import math
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.filters import gabor
from skimage.util import random_noise
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load, Parallel, delayed
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, confusion_matrix, precision_score

# In the end while working, I didn't remember anymore all the relative data I had used while studying
#You can see from the filenames used wich one is included in the process, example: general_functions.create_circular_mask(2)
# Just in case, I kept everything at this point and that point is: the start - me using python for the first time,sorry

# Function to generate masks for zones
def generate_mask_for_zone(zone_file, radius):
    zone = pd.read_pickle(zone_file)  
    data = zone["label_3m"].values.reshape((5000, 5000))
    return create_circular_mask(radius)

def generate_masks_for_zones(list_of_zones, radius):
    """
    Generates a mask for each zone using a circular mask.
    """
    masks = Parallel(n_jobs=-1)(delayed(generate_mask_for_zone)(zone_file, radius) for zone_file in list_of_zones)
    return masks
#optimazed ends

@jit(nopython=True)
def create_circular_mask(radius):
    """
    Creates a fast and efficient circular mask compatible with Dask and large-scale operations.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2:
                kernel[y, x] = 1
    kernel[radius, radius] = 0  # Exclude the center
    return kernel

# Apply circular mask function with NumPy broadcasting for better performance
def apply_circular_mask(data, radius):
    height, width = data.shape
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[-center_y:center_y+1, -center_x:center_x+1]
    mask = (x**2 + y**2 <= radius**2).astype(np.float32)
    return data * mask

def extract_numpy_files_in_folder(path, skip=[]):
     """
     Returns all the .npy files in a given directory.
     Skips subdirectories.
     """
     root, _, files = next(os.walk(path))
     holder = []
        
     for file in files:
         if file[-3:] != ".npy":
             continue
         elif file in skip:
             continue
         holder.append(os.path.join(root,file))
     return holder

def generate_mask(small_radius, big_radius):
     """
     Generate a mask in a circular radius around a point.
     """
     height, width = big_radius*2,big_radius*2
     Y, X = np.ogrid[:height+1, :width+1]
     distance_from_center = np.sqrt((X- big_radius)**2 + (Y-big_radius)**2)
     mask = (small_radius <= distance_from_center) & (distance_from_center <= big_radius)
     return mask

def circle_mask_generator(radius):
     """
     Wrapper around generate_mask for usage when only a circular radius.
     """
     return generate_mask(1,radius)

# Function to generate masks for zones
def generate_mask_for_zone(zone_file, radius):
     zone = pd.read_pickle(zone_file)  
     data = zone["label_3m"].values.reshape((5000, 5000))
     return create_circular_mask(radius)

def create_filter_with_mask(postfix, arr_with_filenames, function, mask):
     """
     Create a filter over an array of filenames.npy files.
     Existing files with correct naming schemes will NOT be updated if they exist.
     _raw files will be skipped.
     Returns an iterator that can be used to show/save filtered arrays. A name is also yielded.
     """
     for filename in arr_with_filenames:
         if filename[-4:] != "_raw":
             continue
         elif os.path.isfile(f"./{filename[:-4]}_{postfix}.npy"):
             continue
         arr = np.load(f"{filename}.npy")
         holder = generic_filter(arr, function, footprint=mask)
         yield (f"{filename[:-4]}_{postfix}", holder)

def merge_numpy_zones_files(list_of_files):
     """
     Takes a list of paths to files and loads them into a panda DataFrame.
     If the name contains the word 'ditches', the name is replaced by 'labels'.
     Only one zone should be contained inside the list.
     """
     holder = {}
     for file in list_of_files:
         if "ditches" in file or "Ditches" in file:
             holder["labels"] = np.load(file).reshape(-1)
         else:
             holder [file.split("/")[-1][:-4]] = np.load(file).reshape(-1)
     return pd.DataFrame(data=holder)

def create_balanced_dataset(list_of_zones, masks):
    """
    Takes a list of zone file paths and generates a dataset based on balanced masks
    from the labels for the different zones.
    """
    frames = np.empty(len(list_of_zones), dtype=object)
    for i, zone_file_name in enumerate(list_of_zones):
        # Load zone and mask
        zone = pd.read_pickle(zone_file_name)
        mask = masks[i]  # Retrieve the mask for the current zone

        print(f"Zone shape before dropping: {zone.shape}")
        print(f"Mask type: {type(mask)}")
        print(f"Mask content preview: {mask[:5]}")
        print(f"Mask shape: {mask.shape}")

        # Flatten and convert mask to Boolean
        if isinstance(mask, np.ndarray) and mask.ndim == 2:  # If 2D array, flatten it
            mask = mask.flatten()
            print(f"Mask flattened to shape: {mask.shape}")

        # Ensure mask is Boolean
        mask = mask.astype(bool)

        print(f"Number of rows to drop (True in mask): {mask.sum()}")

        # Apply mask to drop rows
        try:
            zone.drop(zone.index[mask], inplace=True)
        except Exception as e:
            print(f"Error applying mask: {e}")
            continue

        print(f"Zone shape after dropping: {zone.shape}")
        frames[i] = zone

    # Combine the frames
    frames_list = [frame.shape for frame in frames if frame is not None]
    print(f"Frames list: {frames_list}")
    return pd.concat(frames, ignore_index=True)


# the original function from the original code
def create_balanced_mask(ditchArr, height, width):
     """
     Creates a mask from a labeled zone to balance the ditch and non-ditch classes more.
     """
     new_arr = ditchArr.copy()
     for i in range(0, len(ditchArr), height):
         for j in range(0, len(ditchArr[i]), width):
             zoneContainsDitches = False  # Default to no ditches in this zone
             if random.random() * 100 > 98:
                 zoneContainsDitches = True  # Randomly introduce ditches
             for k in range(height):
                 for l in range(width):
                     if ditchArr[i+k][j+l] == 1:
                         zoneContainsDitches = True
                     if zoneContainsDitches:
                         for m in range(height):
                             for n in range(width):
                                 new_arr[i+m][j+n] = 1
                         break
                 if zoneContainsDitches:
                     break
             if not zoneContainsDitches:
                for m in range(height):
                     for n in range(width):
                         new_arr[i+m][j+n] = 0
     return new_arr

# altered and explained function, not the original one
# Used in the optimisation phase
def create_balanced_mask2(ditchArr, height, width):
    """
    Creates a mask from a labeled zone to balance the ditch and non-ditch classes more.
    """
    new_arr = np.zeros_like(ditchArr, dtype=int)  # Initialize a new array to store the mask
    
    # Iterate over the original array dimensions
    for i in range(ditchArr.shape[0] - height + 1):  # Adjust range for rows, ensuring valid sub-regions
        for j in range(ditchArr.shape[1] - width + 1):  # Adjust range for columns, ensuring valid sub-regions
            
            # Randomly select a column within the range [0, ditchArr.shape[1] - 1] (up to 4999 for a 5000 column array)
            col = random.randint(0, ditchArr.shape[1] - 1)  # Ensure column index is valid (0 to 4999)
            
            zone = ditchArr[i:i+height, j:j+width]  # Extract sub-region
            
            # Check if there's at least one ditch (value == 1) in the sub-region
            if np.any(zone == 1):
                # Randomly decide whether to mark the block with ditches
                if random.random() > 0.98:  # Adjust random threshold
                    new_arr[i, col] = 1  # Mark as ditch at randomly selected column
                else:
                    new_arr[i, col] = 0  # Mark as non-ditch
            else:
                new_arr[i, col] = 0  # Mark as non-ditch if no ditch in zone

    return new_arr


#This phase of extract_features was corrected in discussion with ChatGPT
def extract_features(npy_files, circular_mask_radius=10):
     from Functions.general_functions import some_function  # Delayed import
     # Now you can use some_function in this function

     @jit(nopython=True)
     def apply_circular_mask(data, radius):
         """
         Applies a circular mask to the data to extract features within the radius.
         """
         height, width = data.shape
         mask = np.zeros_like(data, dtype=np.float32)
         center_x, center_y = width // 2, height // 2

         for i in range(height):
             for j in range(width):
                 # Compute distance from the center
                 distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                 if distance <= radius:
                     mask[i, j] = 1  # Include in the mask
         return data * mask

#this "masks" was missing from the original code, this was created in discussion with ChatGPT
def generate_masks_for_zones(list_of_zones, radius):
     """
     Generates a mask for each zone using a circular mask.
     """
     masks = []
    
     for zone_file in list_of_zones:
         # Load the pickle file
         zone = pd.read_pickle(zone_file)  # Load zone data
        
         # Extract the relevant data (assuming "label_3m" contains the data you want to mask)
         data = zone["label_3m"].values.reshape((5000, 5000))  # Adjust shape as needed
        
         # Apply the circular mask
         mask = apply_circular_mask(data, radius)
        
         # Append the mask to the list
         masks.append(mask)
    
     return masks



def yield_training_test_zones(list_of_files):
     folds = len(list_of_files)
     arr = list_of_files.copy()
     for i in range(folds):
       training, testing = np.delete(arr, i, 0), arr[i]
       yield (training, testing)

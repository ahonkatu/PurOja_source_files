#From functions 2.0

# General arr lib
import numpy as np

# Compile py code
from numba import jit, njit
from numba import prange

# Multithreading
import dask.array as da
import dask as dk
from dask import delayed
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
dk.config.set(scheduler='processes')

from dask_image.ndfilters import generic_filter as d_gf

from collections import deque

from skimage.filters import gabor
from skimage.restoration import  denoise_bilateral
from skimage.transform import resize  # For downsampling

from scipy.ndimage import gaussian_filter, generic_filter as gf
from scipy.ndimage import median_filter, uniform_filter 
from scipy.ndimage import generic_filter, binary_closing

import datetime

import numpy as np
from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
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
from joblib import dump, load
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, confusion_matrix, precision_score
from numba import jit
from numba import prange


from . import general_functions
from .general_functions import create_circular_mask

# In the end while working, I didn't remember anymore all the relative data I had used
# Just in case, I kept everything at this point and that point is: the start - me using python for the first time


# Original name, I didn't change it
def create_circular_maskdem(radius):
    """Create a circular boolean mask with given radius"""
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x*x + y*y <= radius*radius
    return mask

def dem_ditch_detection(arr, sensitivity_threshold=-0.08):
    """
    DEM ditch enhancement with adjustable sensitivity threshold.
    """
    arr[arr == -99999] = np.nan  # Replace -99999 with NaN
    newArr = arr.copy()
    maxArr = generic_filter(arr, np.amax, footprint=create_circular_maskdem(30))
    minArr = generic_filter(arr, np.amin, footprint=create_circular_maskdem(10))
    meanArr = generic_filter(arr, np.median, footprint=create_circular_maskdem(10))
    minMaxDiff = arr.copy()

    # Initialize a count variable for detected ditches
    count_ditches = 0

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if minArr[i][j] < maxArr[i][j] - 3:
                minMaxDiff[i][j] = 1
            else:
                minMaxDiff[i][j] = 0

    closing = binary_closing(minMaxDiff, structure=create_circular_maskdem(10))
    closing2 = binary_closing(closing, structure=create_circular_maskdem(10))

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < meanArr[i][j] + sensitivity_threshold:
                newArr[i][j] = meanArr[i][j] - arr[i][j]
                count_ditches += 1  # Increment the count when a ditch is detected
            else:
                newArr[i][j] = 0
            if closing2[i][j] == 1:
                newArr[i][j] = 0

    # Print the count of detected ditches
    print(f"Detected ditches: {count_ditches} pixels")
    return newArr

def sky_view_hpmf_gabor_stream_removal(feature, streamAmp):
    """
    Takes a SkyViewFactor or HPMF feature and a stream amplification to create a new feature with streams weakened.
    """
    conicStreamRemoval = feature.copy()
    maxVal = np.amax(feature)
    for i in range(len(conicStreamRemoval)):
        for j in range(len(conicStreamRemoval[i])):
            print(type(streamAmp))
            print(type(streamAmp[i][j]))
            print(streamAmp)
            if streamAmp[i][j] != 0:
                conicStreamRemoval[i][j] += (streamAmp[i][j] / 2) * maxVal
                if conicStreamRemoval[i][j] > maxVal:
                    conicStreamRemoval[i][j] = maxVal
    return conicStreamRemoval

def impoundment_DEM_stream_strenghten(impFeature, streamAmp):
    """
    Takes a DEM or Impoundment feature and a stream amplification to create a new feature with streams strengthened.
    """
    impStreamStrengthened = impFeature.copy()  # Create a copy to work with
    for i in range(len(impStreamStrengthened)):
        for j in range(len(impStreamStrengthened[i])):
            if streamAmp[i][j] != 0:
                # Strengthen the streams based on streamAmp value
                if streamAmp[i][j] > 0.7:  # Strong streams
                    # Apply a factor greater than 1 (e.g., 1.5 or 2) to strengthen the stream
                    impStreamStrengthened[i][j] = impStreamStrengthened[i][j] * (1 + (streamAmp[i][j] / 2))
                else:
                    # If streamAmp is not strong, apply a smaller strengthening factor (e.g., 1.2)
                    impStreamStrengthened[i][j] = impStreamStrengthened[i][j] * 1.2
    return impStreamStrengthened


def impoundment_dem_stream_removal(impFeature, streamAmp):
    """
    Takes a DEM or Impoundment feature and a stream amplification to create a new feature with streams weakened.
    """
    impStreamRemoval = impFeature.copy()
    for i in range(len(impStreamRemoval)):
        for j in range(len(impStreamRemoval[i])):
            if streamAmp[i][j] != 0:
                impStreamRemoval[i][j] = impStreamRemoval[i][j] * (1 - (streamAmp[i][j] / 2)) if streamAmp[i][j] > 0.7 else impStreamRemoval[i][j] * 0.3
    return impStreamRemoval

def stream_amplification(arr):
    """
    Attempts to amplify only streams with impoundment index.
    """
    streamAmp = arr.copy()
    for i in range(len(streamAmp)):
        for j in range(len(streamAmp[i])):
            if streamAmp[i][j] > 13:
                streamAmp[i][j] = 0
    morphed = morph.grey_dilation(streamAmp, structure = create_circular_mask(35))
    minVal = np.amin(morphed)
    morphed -= minVal
    maxVal = np.amax(morphed)
    morphed /= maxVal if (maxVal != 0) else 1
    return morphed

def _reclassify_impoundment(arr):
    """
    Internally used reclassification of impoundment index with different thresholds.
    """
    new_arr = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == 0:
                new_arr[i, j] = 0
            elif arr[i, j] < 0.003:
                new_arr[i, j] = 5
            elif arr[i, j] < 0.01:
                new_arr[i, j] = 50
            elif arr[i, j] < 0.02:
                new_arr[i, j] = 100
            elif arr[i, j] < 0.04:
                new_arr[i, j] = 1000
            elif arr[i, j] < 0.1:
                new_arr[i, j] = 10000
            elif arr[i, j] < 0.3:
                new_arr[i, j] = 100000
            else:
                new_arr[i, j] = 1000000
    return new_arr

def impoundment_amplification(arr, mask_radius=10):
    """
    Impoundment ditch enhancement.
    """
    # Reclassify the array
    reclassified_arr = _reclassify_impoundment(arr)

     # Convert the array to a Dask array for processing
    norm_arr = da.from_array(reclassified_arr, chunks=(800, 800))

    # Create the mask
    mask = create_circular_mask(mask_radius)

    # Apply the filters
    result = d_gf(
        d_gf(
            d_gf(norm_arr, np.nanmean, footprint=mask),
            np.nanmean, footprint=mask
        ),
        np.nanmedian, footprint=mask
    ).compute(scheduler="processes")

    return result

def compute_mean_min_norm_hpmf(lines, shift_range=5):
    """Compute pixel-wise mean of min norms for better preservation."""
    n_lines = len(lines)
    if n_lines < 2:
        return np.zeros(lines.shape[1])  # Return zero array if not enough rows
    
    min_norms = np.full((n_lines - 1, lines.shape[1]), np.inf)  # Shape correction
    
    for i in range(1, n_lines):
        line1 = lines[i]
        line2 = lines[i - 1]
        
        for shift in range(-shift_range, shift_range + 1):
            shifted_line2 = np.roll(line2, shift)
            norm = np.abs(line1 - shifted_line2)  # Element-wise absolute difference
            min_norms[i - 1] = np.minimum(min_norms[i - 1], norm)
    
    return np.nanmean(min_norms, axis=0)  # Compute column-wise mean

def process_hpmf(hpmf, shift_range=5):
    """Load and process HPMF data without visualization."""
    
    # Ensure float type
    hpmf = hpmf.astype(np.float32)
    
    # Handle missing values
    hpmf = np.where(hpmf == -99999, np.nan, hpmf)
    
    if hpmf.ndim != 2:
        raise ValueError("HPMF data must be a 2D array")
    
    # Initialize the processed array
    hpmf_processed = np.zeros_like(hpmf, dtype=np.float32)

    # Process each row individually
    for i in range(hpmf.shape[0]):
        row_result = compute_mean_min_norm_hpmf(hpmf[max(0, i-5):min(hpmf.shape[0], i+6)], shift_range)
        hpmf_processed[i, :] = row_result  # Assign processed row
    
    # Normalize using percentiles to avoid extreme values
    v_min, v_max = np.nanpercentile(hpmf_processed, [2, 98])  # Robust normalization
    hpmf_processed = (hpmf_processed - v_min) / (v_max - v_min)
    hpmf_processed = np.clip(hpmf_processed, 0, 1)  # Clip to [0,1] range
    
    return hpmf_processed


#original HPMF filter =slow
# @jit(nopython=True)
# def _reclassify_hpmf_filter(arr):
#     """
#     Internally used reclassification of HPMF with different thresholds.
#     """
#     binary = np.copy(arr)
#     for i in range(len(arr)):
#         for j in range(len(arr[i])):
#             if arr[i][j] < 0.000001 and arr[i][j] > -0.000001:
#                 binary[i][j] = 100
#             else:
#                 binary[i][j] = 0
#     return binary


# @jit(nopython=True)
# def _reclassify_hpmf_filter_mean(arr):
#     """
#     Internally used reclassification of HPMF with different thresholds.
#     """
#     reclassify = np.copy(arr)
#     for i in range(len(arr)):
#         for j in range(len(arr[i])):
#             if arr[i][j] < 1:
#                 reclassify[i][j] = 0
#             elif arr[i][j] < 3:
#                 reclassify[i][j] = 1
#             elif arr[i][j] < 7:
#                 reclassify[i][j] = 2
#             elif arr[i][j] < 10:
#                 reclassify[i][j] = 50
#             elif arr[i][j] < 20:
#                 reclassify[i][j] = 75
#             elif arr[i][j] < 50:
#                 reclassify[i][j] = 100
#             elif arr[i][j] < 80:
#                 reclassify[i][j] = 300
#             elif arr[i][j] < 100:
#                 reclassify[i][j] = 600
#             else:
#                 reclassify[i][j] = 1000
#     return reclassify


# @jit
# def hpmf_filter(arr):
#     """
#     HPMF ditch enhancement.
#     """
#     normalized_arr = da.from_array(
#         _reclassify_hpmf_filter(arr), chunks=(800, 800))

#     mean = d_gf(d_gf(d_gf(d_gf(normalized_arr, np.amax, footprint=create_circular_mask(1)), np.amax, footprint=create_circular_mask(
#         1)), np.median, footprint=create_circular_mask(2)), np.nanmean, footprint=create_circular_mask(5)).compute(scheduler='processes')
#     reclassify = da.from_array(
#         _reclassify_hpmf_filter_mean(mean), chunks=(800, 800))

#     return d_gf(reclassify, np.nanmean, footprint=create_circular_mask(7)) #smaller one than in the study

#the idea is to clean up the image with some fluvial features left to see
#edge features for further processing
# Not original ones, altered
#This was no used in the end
# @jit(nopython=True)
# def _reclassify_sky_view_non_ditch_amp(arr):
#     """
#     Internal non ditch amplification reclassification for SkyViewFactor.
#     Idea was to view edge erosion for older streams.
#     In this developmetn stage this was left out.
#     """
#     new_arr = np.copy(arr)
#     for i in range(len(arr)):
#         for j in range(len(arr[i])):
#                 if arr[i, j] < 0.92:
#                     new_arr[i, j] = 46
#                 elif arr[i, j] < 0.935:
#                     new_arr[i, j] = 37
#                 elif arr[i, j] < 0.94:
#                     new_arr[i, j] = 29
#                 elif arr[i, j] < 0.945:
#                     new_arr[i, j] = 22
#                 elif arr[i, j] < 0.95:
#                     new_arr[i, j] = 16
#                 elif arr[i, j] < 0.955:
#                     new_arr[i, j] = 11
#                 elif arr[i, j] < 0.958:
#                     new_arr[i, j] = 7
#                 elif arr[i, j] < 0.963:
#                     new_arr[i, j] = 4
#                 elif arr[i, j] < 0.972:
#                     new_arr[i, j] = 2
#                 else:
#                     new_arr[i, j] = 1
#         return new_arr

# @jit
# def sky_view_non_ditch_amplification(arr):
#     """
#     Non ditch amplification from SkyViewFactor.
#     """
#     arr = da.from_array(arr, chunks=(800, 800))
#     arr = d_gf(arr, np.nanmedian, footprint=create_circular_mask(25)
#                ).compute(scheduler='processes')
#     arr = da.from_array(
#         _reclassify_sky_view_non_ditch_amp(arr), chunks=(800, 800))
#     return d_gf(arr, np.nanmean, footprint=create_circular_mask(10))


@jit
def _sky_view_gabor(merged, gabors):
    """
    Internal SkyViewFactor merge of Gabor filters.
    """
    for i in range(len(merged)):
        for j in range(len(merged[i])):
            merged[i][j] = 0
    for i in range(len(merged)):
        for j in range(len(merged[i])):
            for k in range(len(gabors)):
                merged[i][j] += gabors[k][i][j]
    return merged

def sky_view_gabor(skyViewArr):
    """
    SkyViewFactor Gabor filter.
    """
    delayed_gabors = []
    for i in np.arange(0.03, 0.08, 0.01):
        for j in np.arange(0, 3, 0.52):
            delayed_gabor = dk.delayed(gabor)(skyViewArr, i, j)[0]
            delayed_gabors.append(delayed_gabor)
    gabors = dk.compute(*delayed_gabors)  # Compute all delayed Gabor filters
    merged = _sky_view_gabor(skyViewArr.copy(), gabors)
    return merged


@jit
def sky_view_conic_filter(arr, maskRadius, threshold):
    """
    Ditch amplification by taking the mean of cones in different directions from pixels.
    """
    # Standard values: maskRadius = 5, threshold = 0.975
    masks = []
    for i in range(0, 8):
        masks.append(_create_conic_mask(maskRadius, i))
    new_arr = arr.copy()
    amountOfThresholds = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            values = _mean_from_masks(arr, (i, j), masks)
            dir1 = 2
            dir2 = 2
            dir3 = 2
            dir4 = 2
            if values[0] < threshold and values[4] < threshold:
                dir1 = values[0] if values[0] < values[4] else values[4]
            if values[1] < threshold and values[5] < threshold:
                dir2 = values[1] if values[0] < values[5] else values[4]
            if values[2] < threshold and values[6] < threshold:
                dir3 = values[2] if values[0] < values[6] else values[4]
            if values[3] < threshold and values[7] < threshold:
                dir4 = values[3] if values[0] < values[7] else values[4]
            dir5 = dir1 if dir1 < dir2 else dir2
            dir6 = dir3 if dir3 < dir4 else dir4
            lowest = dir5 if dir5 < dir6 else dir6
            if lowest < threshold:
                amountOfThresholds += 1
                new_arr[i][j] = 0.95 * lowest if lowest * \
                    0.95 < arr[i][j] else arr[i][j]
    return new_arr


@jit(nopython=True)
def _create_conic_mask(radius, direction):
    """
    Create a conic mask for a direction with a certain radius.
    """
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    coords = np.arange(-radius, radius + 1, dtype=np.int32)  # Ensure dtype is int32
    y = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int32)
    x = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int32)

    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            y[i, j] = coords[i]
            x[i, j] = coords[j]

    # Create a mask for each direction
    if direction == 0:  # topright
        mask = np.logical_and(np.greater(x, y), np.less(x, np.abs(y)))
        mask = np.logical_and(mask, np.less_equal(x**2 + y**2, radius**2))
        mask = np.logical_and(mask, np.greater(x, 0))

    elif direction == 1:  # righttop
        mask = np.logical_and(np.greater(x, np.abs(y)), np.less_equal(x**2 + y**2, radius**2))
        mask = np.logical_and(mask, np.less(y, 0))

    elif direction == 2:  # rightbottom
        mask = np.logical_and(np.greater(x, np.abs(y)), np.less_equal(x**2 + y**2, radius**2))
        mask = np.logical_and(mask, np.greater(y, 0))

    elif direction == 3:  # bottomright
        mask = np.logical_and(np.less(np.abs(x), y), np.less_equal(x**2 + y**2, radius**2))
        mask = np.logical_and(mask, np.greater(x, 0))

    elif direction == 4:  # bottomleft
        mask = np.logical_and(np.less(np.abs(x), y), np.less_equal(x**2 + y**2, radius**2))
        mask = np.logical_and(mask, np.less(x, 0))

    elif direction == 5:  # leftbottom
        mask = np.logical_and(np.greater(np.abs(x), np.abs(y)), np.less(x, np.abs(y)))
        mask = np.logical_and(mask, np.less_equal(x**2 + y**2, radius**2))
        mask = np.logical_and(mask, np.greater(y, 0))

    # Manually assign values based on the mask (avoid using `kernel[mask] = 1`)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if mask[i, j]:
                kernel[i, j] = 1

    return kernel

@jit(nopython=True)
def _mean_from_masks(arr, position, masks):
    i, j = position
    radius = (masks.shape[1] - 1) // 2
    values = []

    for mask in masks:
        masked_sum = 0
        masked_count = 0

        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < arr.shape[0] and 0 <= nj < arr.shape[1]:
                    if mask[di + radius, dj + radius] == 1:
                        masked_sum += arr[ni, nj]
                        masked_count += 1

        mean_value = masked_sum / masked_count if masked_count > 0 else np.inf
        values.append(mean_value)

    return values

#@jit(nopython=True, debug=True)
def conic_mean(arr, mask_radius=6, threshold=0.2):
    # Function implementation
    """
    Applies directional mean calculation with a conic mask to the array.
    """
    masks = np.zeros((6, 2 * mask_radius + 1, 2 * mask_radius + 1), dtype=np.float64)
    for direction in range(6):
        masks[direction] = _create_conic_mask(mask_radius, direction)

    new_arr = arr.copy()

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            values = _mean_from_masks(arr, (i, j), masks)

            lowest = min(values)

            if lowest < threshold:
                new_arr[i, j] = 0.9894350171089172 * lowest if lowest * 0.9894350171089172 < arr[i, j] else arr[i, j]

    return new_arr

def _slope_stream_ditch_classification(arr):
    """
    Optimized slope classification for identifying streams and ditches.
    Uses more granular classification in low slopes where streams/ditches typically occur.
    """
    conditions = [
        (arr < 2),           # Very flat areas - potential water accumulation
        (arr >= 2) & (arr < 5),    # Gentle slopes - common for streams
        (arr >= 5) & (arr < 8),    # Mild slopes - potential drainage channels
        (arr >= 8) & (arr < 12),   # Moderate slopes - possible ephemeral streams
        (arr >= 12) & (arr < 15),  # Steeper channels
        (arr >= 15) & (arr < 20),  # Steep terrain features
        (arr >= 20) & (arr < 30),  # Very steep terrain
        (arr >= 30) & (arr < 45),  # Extreme slopes
        (arr >= 45)                # Near-vertical features
    ]
    
    # Values optimized for stream/ditch detection
    values = [1, 3, 6, 10, 13, 17, 25, 35, 45]
    
    result = da.zeros_like(arr)
    for cond, value in zip(conditions, values):
        result = da.where(cond, value, result)
    
    return result

def process_slope_for_channels(arr, chunk_size=(1000, 1000), filter_size=15):
    """
    Process slope data to enhance stream and ditch features.
    
    Parameters:
    - arr: Input slope array
    - chunk_size: Dask chunk size for parallel processing
    - filter_size: Size of median filter window (smaller = more detail)
    """
    arr_dask = da.from_array(arr, chunks=chunk_size)
    
    # Apply smaller median filter to preserve narrow features
    arr_filtered = arr_dask.map_blocks(
        lambda block: median_filter(block, size=(filter_size, filter_size), mode='reflect'),
        dtype=arr_dask.dtype
    )
    
    # Classify slopes
    classified = _slope_stream_ditch_classification(arr_filtered)
    
    return classified

#Dask optimised slope functiion
# def _slope_non_ditch_amplification_normalize_dask(arr):
#     """
#     Internal non-ditch amplification reclassification for Slope, optimized for Dask.
#     This function applies a vectorized operation to normalize the array based on conditions.
#     """
#     # Define the conditions and their corresponding values
#     conditions = [
#         (arr < 8),
#         (arr >= 8) & (arr < 9),
#         (arr >= 9) & (arr < 10),
#         (arr >= 10) & (arr < 11),
#         (arr >= 11) & (arr < 13),
#         (arr >= 13) & (arr < 15),
#         (arr >= 15) & (arr < 17),
#         (arr >= 17) & (arr < 19),
#         (arr >= 19) & (arr < 21),
#         (arr >= 21),
#     ]
#     values = [0, 20, 25, 30, 34, 38, 42, 46, 50, 55]

#     # Use dask's where functionality to apply conditions
#     result = da.zeros_like(arr)
#     for cond, value in zip(conditions, values):
#         result = da.where(cond, value, result)

#     return result


# def slope_non_ditch_amplification_dask(arr, chunk_size=(1000, 1000)):
#     """
#     Non-ditch amplification from Slope, optimized for Dask.
#     Processes the array using a median filter and applies reclassification.
#     """
#     # Convert the input array to a Dask array with specified chunk size
#     arr_dask = da.from_array(arr, chunks=chunk_size)

#     # Apply the median filter using a circular mask of radius 35
#     # Note: Dask does not directly support scipy's median filter, so apply it in a delayed manner
#     arr_filtered = arr_dask.map_blocks(
#         lambda block: median_filter(block, size=(35, 35), mode='reflect'),
#         dtype=arr_dask.dtype,
#     )

#     # Normalize the array using the internal normalization function
#     new_arr = _slope_non_ditch_amplification_normalize_dask(arr_filtered)

#     # Return the processed Dask array
#     return new_arr

def classify_natural_streams(hpmf_array, svf_array, base_threshold=0.989435017108972):
    """
    Classify natural streams using HPMF and SVF data
    
    Parameters:
    hpmf_array: np.array - High pass median filter array
    svf_array: np.array - Sky view factor conic mean array
    base_threshold: float - Base threshold value for classification
    
    Returns:
    np.array - Classified natural streams
    """
    # Ensure arrays are same size
    if hpmf_array.shape != svf_array.shape:
        raise ValueError("Input arrays must have same dimensions")
    
    # Create output array
    result = np.copy(hpmf_array)
    
    # Apply directional filters to detect meandering patterns in SVF
    # Sobel filters for edge detection in multiple directions
    sobel_x = ndimage.sobel(svf_array, axis=0)
    sobel_y = ndimage.sobel(svf_array, axis=1)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Apply gaussian smoothing to reduce noise
    smoothed_gradient = ndimage.gaussian_filter(gradient_magnitude, sigma=1.0)
    
    # Create meandering mask
    meandering_mask = smoothed_gradient > np.mean(smoothed_gradient)
    
    # Process arrays
    for i in range(1, hpmf_array.shape[0] - 1):
        for j in range(1, hpmf_array.shape[1] - 1):
            # Get local minimum SVF value
            local_svf = svf_array[i-1:i+2, j-1:j+2]
            lowest = np.min(local_svf)
            
            # Apply threshold condition
            if lowest < base_threshold:
                new_value = 0.989435017108972 * lowest
                if new_value < svf_array[i,j]:
                    # Only modify values where meandering patterns are detected
                    if meandering_mask[i,j]:
                        result[i,j] = new_value
                    
    # Apply connectivity filter to ensure continuity
    labeled_array, num_features = ndimage.label(result > 0)
    
    # Remove small disconnected segments
    for label in range(1, num_features + 1):
        if np.sum(labeled_array == label) < 50:  # Minimum size threshold
            result[labeled_array == label] = 0
            
    return result

def enhance_stream_features(classified_streams, enhancement_factor=1.5):
    """
    Enhance detected stream features
    
    Parameters:
    classified_streams: np.array - Classified stream array
    enhancement_factor: float - Factor to enhance stream features
    
    Returns:
    np.array - Enhanced stream features
    """
    # Apply contrast enhancement to stream features
    enhanced = np.copy(classified_streams)
    stream_mask = classified_streams > 0
    
    if np.any(stream_mask):
        enhanced[stream_mask] = np.clip(
            enhanced[stream_mask] * enhancement_factor,
            0,
            np.max(classified_streams)
        )
    
    return enhanced
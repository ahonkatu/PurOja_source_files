#From Functions 1.0, note the changes to feature_reation2.py
#feature_creation.py

import numpy as np
from numba import jit, njit
import dask.array as da
from dask_image.ndfilters import generic_filter as d_gf
from skimage.morphology import dilation  # Import the updated dilation function
from skimage.morphology import disk  # Structuring element helper
import rasterio
import os
from scipy.ndimage import binary_dilation
from skimage.morphology import binary_closing
from scipy.ndimage import generic_filter
from dask import delayed, compute  # Import Dask functions
from skimage.filters import gabor

# In the end while working, I didn't remember anymore all the relative data I had used
# Just in case, I kept everything at this point and that point is: the start - me using python for the first time

# Circular Mask Function
# Updated function with the help of ChatGPT.
@njit
def create_circular_mask(radius):
    """
    Creates a circular mask.
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

#Remember to check out the values according the ones you are using.

def _create_conic_mask(image_shape, center, radius, angle_range):
    """
    Creates a conic mask that defines a region-of-interest in the form of a cone.
    - `image_shape`: tuple of (height, width) for the size of the image.
    - `center`: the (y, x) coordinates of the center of the cone.
    - `radius`: the radius of the cone.
    - `angle_range`: tuple defining the angular range (start_angle, end_angle) in degrees.
    """
    # Create a blank mask (all zeros)
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Create a grid of coordinates for the image
    y, x = np.indices(image_shape)
    
    # Calculate the distance from each pixel to the center
    dist_to_center = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    
    # Calculate the angle for each pixel relative to the center (in degrees)
    angle = np.degrees(np.arctan2(y - center[0], x - center[1]))
    
    # Normalize angles to a [0, 360) range
    angle = (angle + 360) % 360
    
    # Define the conic mask using the distance and angle
    is_in_cone = (dist_to_center <= radius) & (angle >= angle_range[0]) & (angle <= angle_range[1])
    
    # Apply the mask to the output
    mask[is_in_cone] = 1
    
    return mask


def extract_features_with_mask(image_dict, label_mask, circular_mask_radius=None):
    """
    Extracts statistical features from images using a binary mask 
    and optionally a circular mask.
    """
    feature_dict = {}
    
    # Create circular mask once if needed
    if circular_mask_radius is not None:
        circular_mask = create_circular_mask(circular_mask_radius)
    
    for file_name, data in image_dict.items():
        # Apply binary mask directly (avoids the need to multiply element-wise in each iteration)
        masked_data = data * label_mask
        
        # Apply optional circular mask in a single step
        if circular_mask_radius is not None:
            masked_data = apply_circular_mask(masked_data, circular_mask_radius)

        # Filter out non-masked areas for statistics
        valid_data = masked_data[masked_data > 0]

        if valid_data.size > 0:  # Ensure there is valid data to compute statistics
            features = {
                "min": np.min(valid_data),
                "max": np.max(valid_data),
                "mean": np.mean(valid_data),
                "median": np.median(valid_data),
                "std_dev": np.std(valid_data),
            }
        else:
            features = {  # If no valid data, set NaN or 0 as placeholders
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std_dev": np.nan,
            }

        feature_dict[file_name] = features
        
    return feature_dict

# Efficient extraction of statistical features using a circular mask
def extract_features(npy_files, circular_mask_radius=10):
    feature_dict = {}
    for file_name, data in npy_files.items():
        masked_data = apply_circular_mask(data, circular_mask_radius)
        features = {
            "min": np.min(masked_data),
            "max": np.max(masked_data),
            "mean": np.mean(masked_data),
            "median": np.median(masked_data),
            "std_dev": np.std(masked_data),
        }
        feature_dict[file_name] = features
    return feature_dict

def _mean_from_masks(image_data, pixel_pos, masks):
    i, j = pixel_pos
    mean_values = []
    
    for mask in masks:
        mask_h, mask_w = mask.shape
        # Compute boundaries for the region
        top = i - mask_h // 2
        bottom = i + mask_h // 2 + 1
        left = j - mask_w // 2
        right = j + mask_w // 2 + 1
        
        # Extract the region and apply the mask
        region = image_data[top:bottom, left:right]
        masked_region = region * mask
        
        # Compute the mean of valid values
        valid_data = masked_region[masked_region > 0]
        mean_values.append(valid_data.mean() if valid_data.size > 0 else 0)
    
    return mean_values


# Vectorized version of reclassification functions
def _reclassify(arr, thresholds, values):
    return np.digitize(arr, thresholds, right=True) * values

# Optimized reclassification functions
def _reclassify_hpmf(arr):
    thresholds = [1, 3, 7, 10, 20, 50, 80, 100]
    values = [0, 1, 2, 50, 75, 100, 300, 600, 1000]
    return _reclassify(arr, thresholds, values)

def _reclassify_impoundment(arr):
    thresholds = [0, 0.002, 0.005, 0.02, 0.05, 0.1, 0.3]
    values = [5, 50, 100, 1000, 10000, 100000, 1000000]
    return _reclassify(arr, thresholds, values)

# Process the features with dask
def impoundment_amplification(arr, mask_radius=10):
    norm_arr = da.from_array(_reclassify_impoundment(arr), chunks=(800, 800))
    mask = create_circular_mask(mask_radius)
    return d_gf(d_gf(d_gf(norm_arr, np.nanmean, footprint=mask), np.nanmean, footprint=mask), np.nanmedian, footprint=mask).compute(scheduler='processes')

# Optimized version for processing multiple images
def process_images(image_folder, label_mask_path, circular_mask_radius=None):
    label_mask = np.load(label_mask_path)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.tif')]
    image_dict = {f: load_tif_as_array(os.path.join(image_folder, f)) for f in image_files}
    return extract_features_with_mask(image_dict, label_mask, circular_mask_radius)

#Numpy array
#def dem_ditch_detection(arr):
 #   max_arr = d_gf(arr, np.amax, footprint=create_circular_mask(30))
  #  min_arr = d_gf(arr, np.amin, footprint=create_circular_mask(10))
   # mean_arr = d_gf(arr, np.median, footprint=create_circular_mask(10))
#    
#    min_max_diff = (min_arr < max_arr - 3).astype(np.float32)
#    closing = morph.binary_closing(min_max_diff, structure=create_circular_mask(10))
#    closing2 = morph.binary_closing(closing, structure=create_circular_mask(10))
    
#    new_arr = np.where((arr < mean_arr - 0.1) & (closing2 == 1), 0, arr - mean_arr)
#    return new_arr

#dask array

def dem_ditch_detection(arr):
    # Apply the generic filters
    max_arr = generic_filter(arr, np.amax, size=(30, 30))
    min_arr = generic_filter(arr, np.amin, size=(10, 10))
    mean_arr = generic_filter(arr, np.median, size=(10, 10))
    
    min_max_diff = (min_arr < max_arr - 3).astype(np.float32)
    
    # Replace 'structure' with 'footprint'
    closing = binary_closing(min_max_diff, footprint=create_circular_mask(10))
    closing2 = binary_closing(closing, footprint=create_circular_mask(10))
    
    # Apply thresholding and difference operation
    new_arr = np.where((arr < mean_arr - 0.1) & (closing2 == 1), 0, arr - mean_arr)
    return new_arr

# Reclassification and morphological operations
#Updated from python 3.7
def stream_amplification(arr):
    """
    Reclassification and morphological operations for stream amplification.
    """
    # Reclassify the array: values less than 14 become 1, others become 0
    stream_amp = (arr < 14).astype(np.float32)  # Ensure float array for further arithmetic operations

    # Create a circular structuring element with radius 35
    structuring_element = disk(35)  # Replace create_circular_mask with skimage's built-in `disk`

    # Apply binary dilation
    morphed = binary_dilation(stream_amp, structure=structuring_element)

    # Ensure the morphed array is float
    morphed = morphed.astype(np.float32)

    # Normalize: subtract the minimum and scale to 0-1
    morphed -= np.amin(morphed)  # Subtract minimum value
    max_val = np.amax(morphed)
    if max_val != 0:  # Avoid division by zero
        morphed /= max_val

    return morphed

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

@jit(nopython=True)
def _reclassify_impoundment(arr):
    """
    Internally used reclassification of impoundment index with different thresholds.
    """
    new_arr = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if new_arr[i, j] == 0:
                new_arr[i, j] = 0
            elif new_arr[i, j] < 0.002:
                new_arr[i, j] = 5
            elif arr[i, j] < 0.005:
                new_arr[i, j] = 50
            elif arr[i, j] < 0.02:
                new_arr[i, j] = 100
            elif arr[i, j] < 0.05:
                new_arr[i, j] = 1000
            elif arr[i, j] < 0.1:
                new_arr[i, j] = 10000
            elif arr[i, j] < 0.3:
                new_arr[i, j] = 100000
            else:
                new_arr[i, j] = 1000000
    return new_arr


#@njit(nopython=False) #enabled as a test because this is not how to use the njit
def impoundment_amplification(arr, mask_radius=10):
    """
    Impoundment ditch enhancement.
    """
    norm_arr = da.from_array(_reclassify_impoundment(arr), chunks=(800, 800))
    mask = create_circular_mask(mask_radius)
    return d_gf(d_gf(d_gf(norm_arr, np.nanmean, footprint=mask), np.nanmean, footprint=mask), np.nanmedian, footprint=mask).compute(scheduler='processes')


@jit(nopython=True)
def _reclassify_hpmf_filter(arr):
    """
    Internally used reclassification of HPMF with different thresholds.
    """
    binary = np.copy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 0.000001 and arr[i][j] > -0.000001:
                binary[i][j] = 100
            else:
                binary[i][j] = 0
    return binary


@jit(nopython=True)
def _reclassify_hpmf_filter_mean(arr):
    """
    Internally used reclassification of HPMF with different thresholds.
    """
    reclassify = np.copy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 1:
                reclassify[i][j] = 0
            elif arr[i][j] < 3:
                reclassify[i][j] = 1
            elif arr[i][j] < 7:
                reclassify[i][j] = 2
            elif arr[i][j] < 10:
                reclassify[i][j] = 50
            elif arr[i][j] < 20:
                reclassify[i][j] = 75
            elif arr[i][j] < 50:
                reclassify[i][j] = 100
            elif arr[i][j] < 80:
                reclassify[i][j] = 300
            elif arr[i][j] < 100:
                reclassify[i][j] = 600
            else:
                reclassify[i][j] = 1000
    return reclassify


@jit
def hpmf_filter(arr):
    """
    HPMF ditch enhancement.
    """
    normalized_arr = da.from_array(
        _reclassify_hpmf_filter(arr), chunks=(800, 800))

    mean = d_gf(d_gf(d_gf(d_gf(normalized_arr, np.amax, footprint=create_circular_mask(1)), np.amax, footprint=create_circular_mask(
        1)), np.median, footprint=create_circular_mask(2)), np.nanmean, footprint=create_circular_mask(5)).compute(scheduler='processes')
    reclassify = da.from_array(
        _reclassify_hpmf_filter_mean(mean), chunks=(800, 800))

    return d_gf(reclassify, np.nanmean, footprint=create_circular_mask(7))

#Updated with the help of ChatGPT (Python versions)
def sky_view_non_ditch_amplification(arr):
    """
    Non-ditch amplification from SkyViewFactor.
    """
    # Ensure `arr` is a Dask array
    arr = da.from_array(arr, chunks=(800, 800))
    
    # Apply median filtering (outside Numba)
    filtered = d_gf(arr, np.nanmedian, footprint=create_circular_mask(25))
    filtered = filtered.compute(scheduler='processes')  # Compute Dask operations
    
    # Reclassify using a separate function
    reclassified = _reclassify_sky_view_non_ditch_amp(filtered)
    
    # Convert back to Dask array and apply mean filter
    reclassified_dask = da.from_array(reclassified, chunks=(800, 800))
    return d_gf(reclassified_dask, np.nanmean, footprint=create_circular_mask(10))

#simple conic_mean
#def sky_view_conic_filter(arr, mask_radius, threshold):
#    masks = [_create_conic_mask(mask_radius, i) for i in range(8)]
#    return np.array([_mean_from_masks(arr, (i, j), masks) for i in range(len(arr)) for j in range(len(arr[i]))])

#For each pixel (i, j), it compares values from different directions and calculates a lowest value.
#If this lowest value is below a specified threshold, the pixel value is modified using the formula 
# #new_arr[i][j] = 0.95 * lowest if lowest * 0.95 < arr[i][j] else arr[i][j]. This appears to "amplify" #the pixel value depending on the conic mean calculation.
@jit
def sky_view_conic_filter(arr, mask_radius, threshold):
    """
    Ditch amplification by taking the mean of cones in different directions from pixels.
    """
    masks = []
    
    # Get image shape and center
    image_shape = arr.shape
    center = (image_shape[0] // 2, image_shape[1] // 2)
    
    # Create conic masks in different directions (8 directions)
    for i in range(8):
        # Define the angle range for each direction (e.g., 0-45 degrees, 45-90 degrees, etc.)
        angle_range = (i * 45, (i + 1) * 45)
        
        # Call _create_conic_mask with the correct arguments
        masks.append(_create_conic_mask(image_shape, center, mask_radius, angle_range))
    
    new_arr = arr.copy()
    amountOfThresholds = 0
    
    # Loop through each pixel in the array
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            # Pass the current pixel position (i, j) along with the image data and masks
            values = _mean_from_masks(arr, (i, j), masks)  # Corrected call to _mean_from_masks
            
            # Initialize direction values
            dir1, dir2, dir3, dir4 = 2, 2, 2, 2
            
            # Compare the values from different directions and select the smallest
            if values[0] < threshold and values[4] < threshold:
                dir1 = min(values[0], values[4])
            if values[1] < threshold and values[5] < threshold:
                dir2 = min(values[1], values[5])
            if values[2] < threshold and values[6] < threshold:
                dir3 = min(values[2], values[6])
            if values[3] < threshold and values[7] < threshold:
                dir4 = min(values[3], values[7])
            
            dir5 = min(dir1, dir2)
            dir6 = min(dir3, dir4)
            lowest = min(dir5, dir6)
            
            # Apply amplification if the lowest value is below the threshold
            if lowest < threshold:
                amountOfThresholds += 1
                new_arr[i][j] = min(0.95 * lowest, arr[i][j])  # Amplify based on the lowest value
    
    return new_arr


#@njit
#def _sky_view_gabor(merged, gabors):
#    """
#    Internal SkyViewFactor merge of Gabor filters.
#    """
#    merged = merged.astype(np.float32)
#    gabors = gabors.astype(np.float32)

    # Reset merged array
#    for i in range(merged.shape[0]):
#        for j in range(merged.shape[1]):
#            merged[i, j] = 0.0

    # Add Gabor filter contributions
#    for k in range(gabors.shape[0]):  # Iterate over filters
#        for i in range(merged.shape[0]):  # Iterate over rows
#            for j in range(merged.shape[1]):  # Iterate over cols
#                merged[i, j] += gabors[k, i, j]

#    return merged


def sky_view_gabor(skyViewArr):
    """
    SkyViewFactor gabor filter.
    """
    delayed_gabors = []
    for i in np.arange(0.03, 0.08, 0.01):
        for j in np.arange(0, 3, 0.52):
            delayed_gabor = da.delayed(gabor)(skyViewArr, i, j)[0]
            delayed_gabors.append(delayed_gabor)
    gabor = da.compute(delayed_gabors)
    return sky_view_gabor(skyViewArr.copy(), gabor[0])

#def sky_view_gabor(skyViewArr):
    """
#    SkyViewFactor gabor filter.
#    """
#    delayed_gabors = []
    
    # Loop through Gabor filter parameters
#    for i in np.arange(0.03, 0.08, 0.01):  # Frequency range
#        for j in np.arange(0, 3, 0.52):    # Orientation range
            # Corrected: delay the Gabor function
#            delayed_gabor = delayed(gabor)(skyViewArr, i, j)[0]
#            delayed_gabors.append(delayed_gabor)
    
    # Compute all delayed tasks in parallel
 #   gabors = compute(*delayed_gabors)
    
    # Process and return the result
#    return sky_view_gabor(skyViewArr.copy(), gabors[0])

@jit
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

@jit(nopython=True)
def _reclassify_sky_view_non_ditch_amp(arr):
    """
    Internal non ditch amplification reclassification for SkyViewFactor.
    """
    new_arr = np.copy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i, j] < 0.92:
                new_arr[i, j] = 46
            elif arr[i, j] < 0.93:
                new_arr[i, j] = 37
            elif arr[i, j] < 0.94:
                new_arr[i, j] = 29
            elif arr[i, j] < 0.95:
                new_arr[i, j] = 22
            elif arr[i, j] < 0.96:
                new_arr[i, j] = 16
            elif arr[i, j] < 0.97:
                new_arr[i, j] = 11
            elif arr[i, j] < 0.98:
                new_arr[i, j] = 7
            elif arr[i, j] < 0.985:
                new_arr[i, j] = 4
            elif arr[i, j] < 0.99:
                new_arr[i, j] = 2
            else:
                new_arr[i, j] = 1
    return new_arr

def load_tif_as_array(file_path):
    """
    Loads a .tif file as a NumPy array.
    
    Parameters:
    - file_path: str, path to the .tif file to be loaded.
    
    Returns:
    - numpy.ndarray: The first band of the image as a 2D array.
    
    Raises:
    - FileNotFoundError: If the file does not exist at the provided path.
    - rasterio.errors.RasterioIOError: If the file is not a valid raster file.
    """
    try:
        with rasterio.open(file_path) as src:
            return src.read(1)  # Returns the first band
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except rasterio.errors.RasterioIOError:
        raise ValueError(f"The file at {file_path} could not be opened as a raster file.")
    
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

@jit(nopython=True)
def _slope_non_ditch_amplification_normalize(arr, new_arr):
    """
    Internal non ditch amplification reclassification for Slope.
    """
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 8:
                new_arr[i][j] = 0
            elif arr[i][j] < 9:
                new_arr[i][j] = 20
            elif arr[i][j] < 10:
                new_arr[i][j] = 25
            elif arr[i][j] < 11:
                new_arr[i][j] = 30
            elif arr[i][j] < 13:
                new_arr[i][j] = 34
            elif arr[i][j] < 15:
                new_arr[i][j] = 38
            elif arr[i][j] < 17:
                new_arr[i][j] = 42
            elif arr[i][j] < 19:
                new_arr[i][j] = 46
            elif arr[i][j] < 21:
                new_arr[i][j] = 50
            else:
                new_arr[i][j] = 55
    return new_arr

@jit
def slope_non_ditch_amplification(arr):
    """
    Non ditch amplification from Slope.
    """
    new_arr = arr.copy()
    arr = d_gf(da.from_array(arr, chunks=(800, 800)), np.nanmedian,
               footprint=create_circular_mask(35)).compute(scheduler='processes')
    new_arr = _slope_non_ditch_amplification_normalize(arr, new_arr)  # Corrected spelling
    return d_gf(da.from_array(new_arr, chunks=(800, 800)), np.nanmean, footprint=create_circular_mask(15))

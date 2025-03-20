# zarr_processing.py
import logging
import zarr
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (cohen_kappa_score, accuracy_score, recall_score,
                           precision_score, f1_score, roc_auc_score, confusion_matrix)
import multiprocessing
import psutil
import concurrent.futures


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 02 RF basic 
def process_zarr_chunk(chunk_key, zarr_file):
    try:
        root = zarr.open(zarr_file, mode='r')
        zone_group = root[chunk_key]
        
        # Force a copy and ensure writability
        data = {
            column: np.array(zone_group[column][:], copy=True).astype(np.float32) 
            for column in zone_group.array_keys()
        }
        
        df = pd.DataFrame(data)
        # Ensure the DataFrame's underlying arrays are writable
        for col in df.columns:
            df[col] = df[col].values.copy()
        
        return df
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_key}: {e}")
        return None

# use with all features, x added to disturb the use when not in use
def load_data_parallel_x(zarr_file, max_workers=3):
    """Load Zarr data efficiently using threading to minimize RAM overhead."""
    with zarr.open(zarr_file, mode='r') as root:
        chunk_keys = list(root.group_keys())

    # Use 'loky' backend which handles spawning better
    dfs = joblib.Parallel(n_jobs=max_workers, backend='loky')(
        joblib.delayed(process_zarr_chunk)(key, zarr_file) for key in chunk_keys
    )
    
    dfs = [df for df in dfs if df is not None]

    # Ensure the final DataFrame is writable
    return pd.concat(dfs, ignore_index=True).copy()

#use with selected features
def load_data_parallel(zarr_file, max_workers=2, features=None):
    """
    Load data from a Zarr file in parallel.
    
    Args:
        zarr_file (str): Path to the Zarr file
        max_workers (int): Maximum number of workers for parallel processing
        features (list): List of features to load, if None, loads all features
    """
    # Open Zarr file
    root = zarr.open(zarr_file, mode='r')
    data_group = root['data']
    
    # Get all available columns
    all_columns = list(data_group.keys())
    
    # Filter columns if features is specified
    if features is not None:
        # Ensure all requested features exist
        missing_features = [f for f in features if f not in all_columns]
        if missing_features:
            raise ValueError(f"The following features are not in the Zarr file: {missing_features}")
        
        # Use only the requested features
        columns_to_load = [col for col in all_columns if col in features]
    else:
        columns_to_load = all_columns
    
    # Load data in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary of futures
        future_to_column = {
            executor.submit(process_zarr_chunk, data_group, col): col 
            for col in columns_to_load
        }
        
        # Process results as they come in
        data_dict = {}
        for future in concurrent.futures.as_completed(future_to_column):
            column = future_to_column[future]
            try:
                data_dict[column] = future.result()
            except Exception as exc:
                print(f'Column {column} generated an exception: {exc}')
                raise
    
    # Convert to DataFrame
    df = pd.DataFrame(data_dict)
    return df

def process_zarr_chunk2(chunk_key, zarr_file):
    try:
        logger.info(f"Starting to process chunk: {chunk_key}")
        root = zarr.open(zarr_file, mode='r')
        zone_group = root[chunk_key]
        
        # Get array keys directly from the zone group
        array_keys = list(zone_group.keys())
        
        # Separate feature arrays and coordinate arrays
        feature_keys = [k for k in array_keys if k not in ['x_coord', 'y_coord']]
        coord_keys = [k for k in array_keys if k in ['x_coord', 'y_coord']]
        
        # Load feature data
        feature_data = {}
        for column in feature_keys:
            try:
                array_data = zone_group[column][:]
                feature_data[column] = np.array(array_data, copy=True).astype(np.float32)
            except Exception as e:
                logger.error(f"Error loading array '{column}': {e}")
        
        if not feature_data:
            logger.error("No feature data was loaded")
            return None
        
        # Create main DataFrame
        df = pd.DataFrame(feature_data)
        
        # Add row and column indices if they exist
        if 'row_idx' in feature_keys and 'col_idx' in feature_keys:
            # These are already in the feature data
            pass
        
        logger.info(f"Created DataFrame with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_key}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# SMOTE 02_1
def load_data_parallel_2(zarr_file, max_workers=3, features=None):
    """
    Load data from a Zarr file in parallel, handling the direct array structure.
    """
    logger.info(f"Opening Zarr file: {zarr_file}")
    root = zarr.open(zarr_file, mode='r')
    
    # Get all zone keys
    chunk_keys = list(root.group_keys())
    logger.info(f"Found chunk keys: {chunk_keys}")
    
    if not chunk_keys:
        raise ValueError(f"No zone groups found in Zarr file: {zarr_file}")
    
    # Process each chunk sequentially for better error handling
    dfs = []
    for chunk_key in chunk_keys:
        logger.info(f"Processing zone: {chunk_key}")
        df = process_zarr_chunk2(chunk_key, zarr_file)
        if df is not None:
            if features is not None:
                # Check if all requested features exist
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    logger.warning(f"Missing features in zone {chunk_key}: {missing_features}")
                    continue
                
                # Filter to requested features
                df = df[features]
                
            dfs.append(df)
        else:
            logger.error(f"Failed to process zone: {chunk_key}")
    
    # Combine all dataframes
    if not dfs:
        raise ValueError("No valid data was loaded from any zone")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    return combined_df
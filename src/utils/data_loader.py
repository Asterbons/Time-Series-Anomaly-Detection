"""Data loading utilities for time series anomaly detection."""

import os
import fnmatch
import zipfile
import numpy as np
import pandas as pd


def read_series(file: str, locations: pd.DataFrame, zf: zipfile.ZipFile, 
                folder_in_zip: str = "phase_1/") -> tuple:
    """
    Load a single time series from a ZIP file.
    
    Args:
        file: Filename within the ZIP (e.g., "001_Anomaly_5000.csv")
        locations: DataFrame with anomaly locations indexed by Name
        zf: Open ZipFile object
        folder_in_zip: Path prefix inside the ZIP
        
    Returns:
        Tuple of (file_name, test_start, data, anomaly)
        - file_name: Base name without extension
        - test_start: Index where test segment begins
        - data: Numpy array of time series values
        - anomaly: Tuple (start, end) or (-1, -1) if unknown
    """
    internal_name = folder_in_zip + file

    with zf.open(internal_name) as f:
        data = pd.read_csv(f, header=None)

    data = np.array(data).flatten()

    # Extract file name components
    file_name = file.split('.')[0]
    splits = file_name.split('_')
    test_start = int(splits[-1])

    # Extract anomaly location if available
    anomaly = (-1, -1)
    if file_name in locations.index:
        row = locations.loc[file_name]
        anomaly = (row["Start"], row["End"])

    return (file_name, test_start, data, anomaly)


def sliding_window(a: np.ndarray, window: int) -> np.ndarray:
    """
    Create sliding window view of an array.
    
    Args:
        a: 1D input array
        window: Window size
        
    Returns:
        2D array where each row is a window of the input
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def load_dataset(zip_path: str = "phase_1.zip", 
                 labels_path: str = "labels.csv",
                 folder_in_zip: str = "phase_1/",
                 limit: int = None) -> tuple:
    """
    Load the complete anomaly detection dataset.
    
    Args:
        zip_path: Path to the ZIP file containing time series
        labels_path: Path to the CSV file with anomaly labels
        folder_in_zip: Path prefix inside the ZIP
        limit: Maximum number of datasets to load (None for all)
        
    Returns:
        Tuple of (file_list, locations, zf, folder_in_zip)
    """
    # Load labels
    locations = pd.read_csv(labels_path)
    locations.set_index("Name", inplace=True)
    
    # Open ZIP file
    zf = zipfile.ZipFile(zip_path)
    
    # Get list of anomaly CSV files
    file_list = np.sort([
        name[len(folder_in_zip):]
        for name in zf.namelist()
        if name.startswith(folder_in_zip)
           and fnmatch.fnmatch(name, "*.csv")
           and not name.endswith("labels.csv")
    ])
    
    # Apply limit if specified
    if limit is not None:
        file_list = file_list[:limit]
    
    return file_list, locations, zf, folder_in_zip

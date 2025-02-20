"""
get_basic.py

This module contains basic utility functions for the repository.
"""
# Standard library imports
import os
import logging
from typing import Optional

# Third-party imports
import torch

from dotenv import load_dotenv

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device(device_map: Optional[str] = "auto") -> torch.device:
    """
    Returns the device on which the model is to be loaded.
    
    :param device_map: The device on which the model is to be loaded.
    :return: The device on which the model is to be loaded.
    """
    if device_map == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else 
                              "cpu")
    elif device_map == "mps":
        device = torch.device("mps")
    elif device_map == "cuda":
        device = torch.device("cuda")
    elif device_map == "cpu":
        device = torch.device("cpu")
    else:
        logger.error("Invalid device_map: %s. Using 'auto' instead.", device_map)
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else
                              "cpu")
    return device


def get_path(path_name: str) -> str:
    """
    Retrieves the full path for a given key from the .env file.
    
    Parameters:
        basic_path_name (str): The key name for the desired path (e.g., 'PROJECT_PATH', 'DISK_PATH').
    
    Returns:
        str: The path value associated with the given key.
        
    Raises:
        ValueError: If the key is not found in the .env file or the value is empty.
    """
    load_dotenv(override=True)

    path = os.getenv(path_name)
    if not path:
        raise ValueError(f"{path_name} not found in .env file or is empty")
    
    return path
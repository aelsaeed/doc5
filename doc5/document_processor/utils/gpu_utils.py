"""
GPU detection and utilities
"""
import logging
import torch

logger = logging.getLogger(__name__)

def check_gpu_availability():
    """
    Check for GPU availability and return the appropriate device
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
        except Exception as e:
            logger.warning(f"Error getting GPU name: {str(e)}")
    else:
        logger.info("No GPU detected, using CPU")
    
    return device

def get_optimal_batch_size(device):
    """
    Determine optimal batch size based on available device
    
    Args:
        device (torch.device): The device to use for computation
        
    Returns:
        int: Recommended batch size
    """
    if device.type == "cuda":
        # Get GPU memory information
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # Convert bytes to GB
            gpu_mem_gb = gpu_mem / (1024**3)
            
            # Simple heuristic for batch size based on GPU memory
            if gpu_mem_gb > 10:
                return 16
            elif gpu_mem_gb > 6:
                return 8
            else:
                return 4
        except Exception as e:
            logger.warning(f"Error determining GPU memory: {str(e)}")
            return 4
    else:
        # Default batch size for CPU
        return 2

def initialize_model_on_device(model_class, model_name, device=None):
    """
    Initialize a model on the appropriate device
    
    Args:
        model_class: The model class to instantiate
        model_name (str): Name or path of the model
        device (torch.device, optional): Device to load the model on
        
    Returns:
        object: Initialized model on the specified device
    """
    if device is None:
        device = check_gpu_availability()
    
    try:
        model = model_class(model_name)
        model.to(device)
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise
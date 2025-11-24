"""Utilities for Detectron2 model loading and weight management."""

import logging
import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("SpheroidPy.utils.detectron_utils")

# Detectron2 Imports
try:
    import detectron2
    import torch
    
    from detectron2.utils.logger import setup_logger
    setup_logger()
    
    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    
    # supress unnecessary output from Detectron in JupyterNotebook
    logging.getLogger("detectron2").setLevel(logging.ERROR)
    
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    DefaultPredictor = None
    get_cfg = None
    model_zoo = None


class DownloadProgressBar:
    """Progress bar for downloading files."""
    
    def __init__(self):
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
            self.pbar = None
        except ImportError:
            self.tqdm = None
            self.pbar = None
    
    def __enter__(self):
        if self.tqdm:
            self.pbar = self.tqdm(
                unit="B", 
                unit_scale=True, 
                miniters=1, 
                desc="Downloading model weights for Deep-Learning-Based Segmentation"
            )
        return self
    
    def __exit__(self, *args):
        if self.pbar:
            self.pbar.close()
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar.
        
        Args:
            b: Number of blocks transferred
            bsize: Block size
            tsize: Total size (if known)
        """
        if self.pbar:
            if tsize is not None:
                self.pbar.total = tsize
            self.pbar.update(b * bsize - self.pbar.n)


def download_detectron_weights(
    weights_dir: Path, 
    weight_file: str = "detectron_model_final.pth"
) -> Path:
    """Download Detectron2 model weights if they don't exist.
    
    Args:
        weights_dir: Directory where weights should be stored
        weight_file: Name of the weight file to extract
        
    Returns:
        Path to the downloaded weight file
        
    Raises:
        FileNotFoundError: If download or extraction fails
    """
    weight_path = weights_dir / weight_file
    
    # If weights already exist, return path
    if weight_path.exists():
        logger.info(f"Model weights found at: {weight_path}")
        return weight_path
    
    # URL for downloading weights
    url = "https://zenodo.org/record/7552508/files/weights.zip?download=1"
    weights_zip = weights_dir / "weights.zip"
    
    # Create weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)
    
    # Download ZIP file if it doesn't exist
    if not weights_zip.exists():
        logger.info(f"Downloading model weights from: {url}")
        try:
            with DownloadProgressBar() as progress:
                urllib.request.urlretrieve(
                    url, 
                    filename=weights_zip, 
                    reporthook=progress.update_to
                )
            logger.info(f"Downloaded ZIP file: {weights_zip}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to download weights from {url}: {e}")
    
    # Extract only the target file from ZIP
    logger.info(f"Extracting {weight_file} from ZIP...")
    try:
        with zipfile.ZipFile(weights_zip, 'r') as zip_ref:
            # Find the target file in the ZIP
            found = False
            target_file_in_zip = None
            
            # First pass: find the file
            for file in zip_ref.namelist():
                # Check if this file matches our target (handle various path structures)
                if (file.endswith(weight_file) or 
                    os.path.basename(file) == weight_file or
                    file == weight_file):
                    target_file_in_zip = file
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(f"{weight_file} not found in ZIP archive")
            
            # Extract the file
            zip_ref.extract(target_file_in_zip, weights_dir)
            extracted_path = weights_dir / target_file_in_zip
            
            # Move to final location if needed
            final_path = weights_dir / weight_file
            if extracted_path != final_path:
                if extracted_path.exists():
                    # Ensure target directory exists
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    # Move file
                    if final_path.exists():
                        final_path.unlink()  # Remove existing file if any
                    shutil.move(str(extracted_path), str(final_path))
                    # Try to remove empty parent directories
                    try:
                        parent = extracted_path.parent
                        while parent != weights_dir and parent.exists():
                            if not any(parent.iterdir()):  # Empty directory
                                parent.rmdir()
                                parent = parent.parent
                            else:
                                break
                    except OSError:
                        pass  # Ignore errors when removing directories
                extracted_path = final_path
            
            if not final_path.exists():
                raise FileNotFoundError(
                    f"Extracted file not found at expected location: {final_path}"
                )
            
            logger.info(f"Extracted {weight_file} to: {final_path}")
                
    except Exception as e:
        raise FileNotFoundError(f"Failed to extract weights from ZIP: {e}")
    
    # Clean up ZIP file after extraction
    try:
        weights_zip.unlink()
        logger.info(f"Removed temporary ZIP file: {weights_zip}")
    except Exception as e:
        logger.warning(f"Could not remove temporary ZIP file {weights_zip}: {e}")
    
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found after extraction: {weight_path}")
    
    return weight_path


def get_detectron_predictor(weights_dir: Optional[Path] = None) -> DefaultPredictor:
    """Initialize and return a Detectron2 predictor.
    
    Automatically downloads model weights if they don't exist.
    
    Args:
        weights_dir: Optional directory where weights should be stored.
                    If None, uses package-relative path.
    
    Returns:
        DefaultPredictor instance configured for spheroid segmentation
        
    Raises:
        ImportError: If Detectron2 is not installed
        FileNotFoundError: If weights cannot be found or downloaded
    """
    if not DETECTRON2_AVAILABLE:
        raise ImportError(
            "Detectron2 is not installed. Please install it to use AI segmentation."
        )
    
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'  # Set device to CPU
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    
    # Get weights directory (relative to package root if not specified)
    if weights_dir is None:
        # Use package-relative path: SpheroidPy/weights/
        # __file__ is SpheroidPy/utils/detectron_utils.py
        # parent is SpheroidPy/utils/
        # parent.parent is SpheroidPy/
        weights_dir = Path(__file__).parent.parent / 'weights'
    weight_file = 'detectron_model_final.pth'
    
    # Download weights if they don't exist
    try:
        weight_path = download_detectron_weights(weights_dir, weight_file)
    except Exception as e:
        raise FileNotFoundError(
            f"Model weights not found and automatic download failed.\n"
            f"Expected location: {weights_dir / weight_file}\n"
            f"Error: {e}\n"
            f"Please download the weights manually or check your internet connection."
        )
    
    cfg.MODEL.WEIGHTS = str(weight_path)
    logger.info(f"Using model weights from: {weight_path}")
    
    return DefaultPredictor(cfg)


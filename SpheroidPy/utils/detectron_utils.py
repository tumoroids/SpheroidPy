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
        if self.pbar is not None:
            self.pbar.close()
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar.
        
        Args:
            b: Number of blocks transferred
            bsize: Block size
            tsize: Total size (if known)
        """
        if self.pbar is not None:
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
    
    # Extract the best matching weight from ZIP.
    # Historically, some archives ship weights under names like:
    # - model_final.pth (non-IncuCyte)
    # - model_final_Incu.pth (IncuCyte-specific)
    # We always normalize the selected file to `weight_file` on disk.
    logger.info(f"Extracting model weights from ZIP (target name on disk: {weight_file})...")
    try:
        with zipfile.ZipFile(weights_zip, 'r') as zip_ref:
            namelist = list(zip_ref.namelist())

            def _basename(p: str) -> str:
                # Zip paths always use forward slashes.
                return os.path.basename(p.rstrip("/"))

            # Candidate mapping: choose non-IncuCyte if both exist.
            # Order = priority.
            preferred_basenames = [
                weight_file,            # current expected canonical name
                "model_final.pth",       # common non-Incu name in some zips
                "model_final_Incu.pth",  # IncuCyte-specific (fallback only)
            ]

            # Build a list of zip entries whose basename matches one of the candidates
            matches: list[str] = []
            for entry in namelist:
                b = _basename(entry)
                if b in preferred_basenames:
                    matches.append(entry)

            if not matches:
                raise FileNotFoundError(
                    f"Could not find any known Detectron2 weight file in ZIP. "
                    f"Expected one of: {preferred_basenames}. "
                    f"Found examples: {namelist[:25]}"
                )

            # Pick best match by preferred_basenames order (non-Incu first)
            def _rank(entry: str) -> int:
                b = _basename(entry)
                try:
                    return preferred_basenames.index(b)
                except ValueError:
                    return 999

            matches.sort(key=_rank)
            chosen_in_zip = matches[0]
            chosen_base = _basename(chosen_in_zip)

            if chosen_base == "model_final_Incu.pth":
                logger.warning(
                    "Detected only IncuCyte weight ('model_final_Incu.pth') or it was ranked best. "
                    "Proceeding, but ensure this is the intended model."
                )

            # Extract chosen file
            zip_ref.extract(chosen_in_zip, weights_dir)
            extracted_path = weights_dir / chosen_in_zip
            final_path = weights_dir / weight_file

            # Normalize into weights_dir/weight_file (rename/move + overwrite)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                final_path.unlink()
            if extracted_path.exists():
                shutil.move(str(extracted_path), str(final_path))
            else:
                # Some zip implementations may extract with different path handling;
                # fall back to searching by basename in weights_dir.
                fallback = next(weights_dir.rglob(chosen_base), None)
                if fallback is None or not fallback.exists():
                    raise FileNotFoundError(
                        f"Extracted file '{chosen_base}' not found after extraction."
                    )
                shutil.move(str(fallback), str(final_path))

            # Cleanup: remove any now-empty subfolders created by extraction
            try:
                # Remove the original extracted directory tree if present
                parent = (weights_dir / chosen_in_zip).parent
                while parent != weights_dir and parent.exists():
                    if not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                    else:
                        break
            except OSError:
                pass

            # Optional cleanup: if the ZIP also contained the other variant,
            # ensure we don't leave duplicate weight files around.
            for extra_base in ("model_final.pth", "model_final_Incu.pth"):
                extra_path = weights_dir / extra_base
                if extra_path.exists() and extra_path.name != final_path.name:
                    try:
                        extra_path.unlink()
                    except OSError:
                        pass

            if not final_path.exists():
                raise FileNotFoundError(f"Extracted weight not found at expected location: {final_path}")

            logger.info(f"Using Detectron2 weights: {final_path} (from ZIP entry: {chosen_in_zip})")
                
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
        # Use a user-writable shared location consistent with HRNet
        weights_dir = Path.home() / ".spheroidpy" / "models"
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


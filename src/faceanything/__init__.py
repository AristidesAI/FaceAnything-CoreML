"""FaceAnything: 4D Face Reconstruction from Any Image Sequence (inference release).

Public helpers:
    load_model      -- build the model and load the released checkpoint
    run_inference   -- run the model on a clip -> FacePrediction
"""
from .model import load_model
from .predict import run_inference, FacePrediction

__all__ = ["load_model", "run_inference", "FacePrediction"]
__version__ = "1.0.0"

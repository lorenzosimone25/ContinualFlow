"""ContinualFlow's reproducible 2D experiment package.

Public experiment and plotting interfaces live in ``experiment_2d`` and
``plotting``. Keeping package import lightweight avoids loading Matplotlib when
only datasets or models are needed.
"""

from .datasets import Normalization, ToyDataset, load_toy_dataset

__all__ = ["Normalization", "ToyDataset", "load_toy_dataset"]

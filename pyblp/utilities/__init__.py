"""Loads general functionality into the top-level utilities namespace and initializes output."""

from .. import options
from .basics import extract_matrix, extract_size, Matrices, ParallelItems, Output
from .statistics import IV, compute_gmm_se, compute_2sls_weights, compute_gmm_weights


output = Output(options)

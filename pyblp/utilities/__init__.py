"""Loads general functionality into the top-level utilities namespace and initializes output."""

from .. import options
from .statistics import IV, compute_gmm_se, compute_2sls_weights, compute_gmm_weights
from .basics import extract_matrix, extract_size, Matrices, Groups, ParallelItems, Output
from .algebra import (
    multiply_tensor_and_matrix, multiply_matrix_and_tensor, precisely_solve, precisely_invert, approximately_solve,
    approximately_invert
)


output = Output(options)

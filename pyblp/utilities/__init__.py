"""Loads general functionality into the top-level utilities namespace and initializes output."""

from .. import options
from .basics import extract_matrix, extract_size, Matrices, ParallelItems, Output


output = Output(options)

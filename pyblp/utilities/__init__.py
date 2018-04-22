"""Loads general functionality into the top-level utilities namespace and initializes output."""

from .. import options
from .iteration import Iteration
from .integration import Integration
from .optimization import Optimization
from .basics import extract_matrix, Matrices, ParallelItems, Output


output = Output(options)

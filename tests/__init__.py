"""Configures NumPy so that it raises all warnings as exceptions."""

import numpy as np


np.seterr(all='raise')

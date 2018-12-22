"""Data construction."""

from typing import Any, Callable, List, Mapping, Optional

import numpy as np

from . import options
from .configurations.formulation import Formulation
from .utilities.basics import Array, Groups, RecArray, extract_matrix, structure_matrices


def build_id_data(T: int, J: int, F: int) -> RecArray:
    r"""Build a balanced panel of market and firm IDs.

    This function can be used to build ``id_data`` for :class:`Simulation` initialization.

    Parameters
    ----------
    T : `int`
        Number of markets.
    J : `int`
        Number of products in each market.
    F : `int`
        Number of firms. If ``J`` is divisible by ``F``, firms produce ``J / F`` products in each market. Otherwise,
        firms with smaller IDs will produce excess products.

    Returns
    -------
    `recarray`
        IDs that associate products with markets and firms. Each of the ``T * J`` rows corresponds to a product. Fields:

            - **market_ids** : (`object`) - Market IDs that take on values from ``0`` to ``T - 1``.

            - **firm_ids** : (`object`) - Firm IDs that take on values from ``0`` to ``F - 1``.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_id_data.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # validate the counts
    if not isinstance(T, int) or not isinstance(F, int) or T < 1 or F < 1:
        raise ValueError("Both T and F must be positive ints.")
    if not isinstance(J, int) or J < F:
        raise ValueError("J must be an int that is at least F.")

    # build and structure IDs
    return structure_matrices({
        'market_ids': (np.repeat(np.arange(T), J).astype(np.int), np.object),
        'firm_ids': (np.floor(np.tile(np.arange(J), T) * F / J).astype(np.int), np.object)
    })


def build_ownership(product_data: Mapping, kappa_specification: Optional[Callable[[Any, Any], float]] = None) -> Array:
    r"""Build ownership matrices, :math:`O`.

    Ownership matrices are defined by their cooperation matrix counterparts, :math:`\kappa`. For each market :math:`t`,
    :math:`O_{jk} = \kappa_{fg}` where :math:`j \in \mathscr{J}_{ft}`, the set of products produced by firm :math:`f` in
    the market, and similarly, :math:`g \in \mathscr{J}_{gt}`.

    Parameters
    ----------
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms.

    kappa_specification : `callable, optional`
        A function that specifies each market's cooperation matrix, :math:`\kappa`. The function is of the following
        form::

            kappa(f, g) -> value

        where ``value`` is :math:`O_{jk}` and both ``f`` and ``g`` are firm IDs from the ``firm_ids`` field of
        ``product_data``.

        The default specification, ``lambda: f, g: int(f == g)``, constructs traditional ownership matrices. That is,
        :math:`\kappa = I`, the identify matrix, implies that :math:`O_{jk}` is :math:`1` if the same firm produces
        products :math:`j` and :math:`k`, and is :math:`0` otherwise.

        If ``firm_ids`` happen to be indices for an actual :math:`\kappa` matrix, ``lambda f, g: kappa[f, g]`` will
        build ownership matrices according to the matrix ``kappa``.

    Returns
    -------
    `ndarray`
        Stacked :math:`J_t \times J_t` ownership matrices, :math:`O`, for each market :math:`t`. If a market has fewer
        products than others, extra columns will contain ``numpy.nan``.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_ownership.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # extract and validate IDs
    market_ids = extract_matrix(product_data, 'market_ids')
    firm_ids = extract_matrix(product_data, 'firm_ids')
    if market_ids is None:
        raise KeyError("product_data must have a market_ids field.")
    if firm_ids is None:
        raise KeyError("product_data must have a firm_ids field.")
    if market_ids.shape[1] > 1:
        raise ValueError("The market_ids field of product_data must be one-dimensional.")
    if firm_ids.shape[1] > 1:
        raise ValueError("The firm_ids field of product_data must be one-dimensional.")

    # validate or use the default kappa specification
    if kappa_specification is None:
        kappa_specification = lambda f, g: np.where(f == g, 1, 0).astype(options.dtype)
    elif callable(kappa_specification):
        kappa_specification = np.vectorize(kappa_specification, [options.dtype])
    if not callable(kappa_specification):
        raise ValueError("kappa_specification must be None or callable.")

    # determine the overall number of products and the maximum number in a market
    N = market_ids.size
    max_J = np.unique(market_ids, return_counts=True)[1].max()

    # construct the ownership matrices
    ownership = np.full((N, max_J), np.nan, options.dtype)
    for t in np.unique(market_ids):
        ids_t = firm_ids[market_ids.flat == t]
        tiled_ids_t = np.tile(np.c_[ids_t], ids_t.size)
        ownership[market_ids.flat == t, :ids_t.size] = kappa_specification(tiled_ids_t, tiled_ids_t.T)
    return ownership


def build_blp_instruments(formulation: Formulation, product_data: Mapping) -> Array:
    r"""Construct traditional excluded BLP instruments.

    Traditional excluded BLP instruments are

    .. math:: \mathrm{BLP}(X) = [\mathrm{Other}(X), \mathrm{Rival}(X)],

    in which :math:`X` is a matrix of product characteristics, :math:`\mathrm{Rival}(X)` consists of sums over
    characteristics of rival goods, and :math:`\mathrm{Other}(X)` consists of sums over characteristics of other
    non-rival goods. All three matrices have the same dimensions.

    Let :math:`x_{jt}` be the vector of characteristics in :math:`X` for product :math:`j` in market :math:`t`, which is
    produced by firm :math:`f`. That is, :math:`j \in \mathscr{J}_{ft}`. Its counterpart in :math:`\mathrm{Rival}(X)` is

    .. math:: \sum_{r \notin \mathscr{J}_{ft}} x_{rt},

    and its counterpart in :math:`\mathrm{Other}(X)` is

    .. math:: \sum_{r \in \mathscr{J}_{ft} \setminus \{j\}} x_{rt}.

    Parameters
    ----------
    formulation : `Formulation`
        :class:`Formulation` configuration for :math:`X`, the matrix of product characteristics used to build excluded
        instruments. Variable names should correspond to fields in ``product_data``.
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms.

        Along with ``market_ids`` and ``firm_ids``, the names of any additional fields can be used as variables in
        ``formulation``.

    Returns
    -------
    `ndarray`
        Traditional excluded BLP instruments :math:`\mathrm{BLP}(X)`.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_blp_instruments.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # load IDs
    market_ids = extract_matrix(product_data, 'market_ids')
    firm_ids = extract_matrix(product_data, 'firm_ids')
    if market_ids is None or firm_ids is None:
        raise KeyError("product_data must have market_ids and firm_ids fields.")
    if market_ids.shape[1] > 1:
        raise ValueError("The market_ids field of product_data must be one-dimensional.")
    if firm_ids.shape[1] > 1:
        raise ValueError("The firm_ids field of product_data must be one-dimensional.")

    # construct a set of market-firm pair IDs
    paired_ids = market_ids.flatten().astype(np.object)
    paired_ids[:] = list(zip(market_ids, firm_ids))

    # initialize grouping objects
    paired_groups = Groups(paired_ids)
    market_groups = Groups(market_ids)

    # build the instruments
    X = build_matrix(formulation, product_data)
    other = paired_groups.expand(paired_groups.sum(X)) - X
    rival = market_groups.expand(market_groups.sum(X)) - X - other
    return np.ascontiguousarray(np.c_[other, rival])


def build_matrix(formulation: Formulation, data: Mapping) -> Array:
    r"""Construct a matrix according to a formulation.

    Parameters
    ----------
    formulation : `Formulation`
        :class:`Formulation` configuration for the matrix. Variable names should correspond to fields in ``data``.
    data : `structured array-like`
        Fields can be used as variables in ``formulation``.

    Returns
    -------
    `ndarray`
        The built matrix.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_matrix.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """
    if not isinstance(formulation, Formulation):
        raise TypeError("formulation must be a Formulation instance.")
    if formulation._absorbed_terms:
        raise ValueError("formulation does not support fixed effect absorption.")
    return formulation._build_matrix(data)[0]

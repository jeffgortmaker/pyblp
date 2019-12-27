"""Data construction."""

from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

import numpy as np

from . import exceptions, options
from .configurations.formulation import Formulation
from .configurations.integration import Integration
from .utilities.basics import Array, Groups, RecArray, extract_matrix, interact_ids, structure_matrices, get_indices


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
    r"""Construct "sums of characteristics" excluded BLP instruments.

    Traditional "sums of characteristics" BLP instruments are

    .. math:: Z^\text{BLP}(X) = [Z^\text{BLP,Other}(X), Z^\text{BLP,Rival}(X)],

    in which :math:`X` is a matrix of product characteristics, :math:`Z^\text{BLP,Other}(X)` is a second matrix that
    consists of sums over characteristics of non-rival goods, and :math:`Z^\text{BLP,Rival}(X)` is a third matrix that
    consists of sums over rival goods. All three matrices have the same dimensions.

    .. note::

       To construct simpler, firm-agnostic instruments that are sums over characteristics of other goods, specify a
       constant column of firm IDs and keep only the first half of the instrument columns.

    Let :math:`x_{jt}` be the vector of characteristics in :math:`X` for product :math:`j` in market :math:`t`, which is
    produced by firm :math:`f`. That is, :math:`j \in \mathscr{J}_{ft}`. Then,

    .. math::

       Z_{jt}^\text{BLP,Other}(X) = \sum_{k \in \mathscr{J}_{ft} \setminus \{j\}} x_{kt}, \\
       Z_{jt}^\text{BLP,Rival}(X) = \sum_{k \notin \mathscr{J}_{ft}} x_{kt}.

    .. note::

       Usually, any supply or demand shifters are added to these excluded instruments, depending on whether they are
       meant to be used for demand- or supply-side estimation.

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
        Traditional "sums of characteristics" BLP instruments, :math:`Z^\text{BLP}(X)`.

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

    # initialize grouping objects
    market_groups = Groups(market_ids)
    paired_groups = Groups(interact_ids(market_ids, firm_ids))

    # build the instruments
    X = build_matrix(formulation, product_data)
    other = paired_groups.expand(paired_groups.sum(X)) - X
    rival = market_groups.expand(market_groups.sum(X)) - X - other
    return np.ascontiguousarray(np.c_[other, rival])


def build_differentiation_instruments(
        formulation: Formulation, product_data: Mapping, version: str = 'local', interact: bool = False) -> Array:
    r"""Construct excluded differentiation instruments.

    Differentiation instruments in the spirit of :ref:`references:Gandhi and Houde (2017)` are

    .. math:: Z^\text{Diff}(X) = [Z^\text{Diff,Other}(X), Z^\text{Diff,Rival}(X)],

    in which :math:`X` is a matrix of product characteristics, :math:`Z^\text{Diff,Other}(X)` is a second matrix that
    consists of sums over functions of differences between non-rival goods, and :math:`Z^\text{Diff,Rival}(X)` is a
    third matrix that consists of sums over rival goods. Without optional interaction terms, all three matrices have the
    same dimensions.

    .. note::

       To construct simpler, firm-agnostic instruments that are sums over functions of differences between all different
       goods, specify a constant column of firm IDs and keep only the first half of the instrument columns.

    Let :math:`x_{jt\ell}` be characteristic :math:`\ell` in :math:`X` for product :math:`j` in market :math:`t`, which
    is produced by firm :math:`f`. That is, :math:`j \in \mathscr{J}_{ft}`. Then in the "local" version of
    :math:`Z^\text{Diff}(X)`,

    .. math::
       :label: local_instruments

       Z_{jt\ell}^\text{Local,Other}(X) =
       \sum_{k \in \mathscr{J}_{ft} \setminus \{j\}} 1(|d_{jkt\ell}| < \text{SD}_\ell), \\
       Z_{jt\ell}^\text{Local,Rival}(X) =
       \sum_{k \notin \mathscr{J}_{ft}} 1(|d_{jkt\ell}| < \text{SD}_\ell),

    where :math:`d_{jkt\ell} = x_{kt\ell} - x_{jt\ell}` is the difference between products :math:`j` and :math:`k` in
    terms of characteristic :math:`\ell`, :math:`\text{SD}_\ell` is the standard deviation of these pairwise differences
    computed across all markets, and :math:`1(|d_{jkt\ell}| < \text{SD}_\ell)` indicates that products :math:`j` and
    :math:`k` are close to each other in terms of characteristic :math:`\ell`.

    The intuition behind this "local" version is that demand for products is often most influenced by a small number of
    other goods that are very similar. For the "quadratic" version of :math:`Z^\text{Diff}(X)`, which uses a more
    continuous measure of the distance between goods,

    .. math::
       :label: quadratic_instruments

       Z_{jtk}^\text{Quad,Other}(X) = \sum_{k \in \mathscr{J}_{ft} \setminus\{j\}} d_{jkt\ell}^2, \\
       Z_{jtk}^\text{Quad,Rival}(X) = \sum_{k \notin \mathscr{J}_{ft}} d_{jkt\ell}^2.

    With interaction terms, which reflect covariances between different characteristics, the summands for the "local"
    versions are :math:`1(|d_{jkt\ell}| < \text{SD}_\ell) \times d_{jkt\ell'}` for all characteristics :math:`\ell'`,
    and the summands for the "quadratic" versions are :math:`d_{jkt\ell} \times d_{jkt\ell'}` for all
    :math:`\ell' \geq \ell`.

    .. note::

       Usually, any supply or demand shifters are added to these excluded instruments, depending on whether they are
       meant to be used for demand- or supply-side estimation.

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

    version : `str, optional`
        The version of differentiation instruments to construct:

            - ``'local'`` (default) - Construct the instruments in :eq:`local_instruments` that consider only the
              characteristics of "close" products in each market.

            - ``'quadratic'`` - Construct the more continuous instruments in :eq:`quadratic_instruments` that consider
              all products in each market.

    interact : `bool, optional`
        Whether to include interaction terms between different product characteristics, which can help capture
        covariances between product characteristics.

    Returns
    -------
    `ndarray`
        Excluded differentiation instruments, :math:`Z^\text{Diff}(X)`.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_differentiation_instruments.ipynb

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

    # identify markets
    market_indices = get_indices(market_ids)

    # build the matrix and count its dimensions
    X = build_matrix(formulation, product_data)
    N, K = X.shape

    # for the local version, do a first pass to compute standard deviations of pairwise differences across all markets
    sd_mapping: Dict[int, Array] = {}
    if version == 'local':
        for k in range(K):
            distances_count = distances_sum = squared_distances_sum = 0
            for t, indices_t in market_indices.items():
                x = X[indices_t][:, [k]]
                distances = x - x.T
                np.fill_diagonal(distances, np.nan)
                distances_count += distances.size - x.size
                distances_sum += np.nansum(distances)
                squared_distances_sum += np.nansum(distances**2)
            sd_mapping[k] = np.sqrt(squared_distances_sum / distances_count - (distances_sum / distances_count)**2)

    # build instruments market-by-market to conserve memory
    other_blocks: List[List[Array]] = []
    rival_blocks: List[List[Array]] = []
    for t, indices_t in market_indices.items():
        # build distance matrices for all characteristics
        distances_mapping: Dict[int, Array] = {}
        for k in range(K):
            x = X[indices_t][:, [k]]
            distances_mapping[k] = x - x.T
            np.fill_diagonal(distances_mapping[k], np.nan)

        # define a function that generates the terms used to create the instruments
        def generate_instrument_terms() -> Iterator[Array]:
            """Generate terms that will be summed to create instruments."""
            for k1 in range(K):
                if version == 'quadratic':
                    for k2 in range(k1, K if interact else k1 + 1):
                        yield np.nan_to_num(distances_mapping[k1]) * np.nan_to_num(distances_mapping[k2])
                elif version == 'local':
                    with np.errstate(invalid='ignore'):
                        close = np.nan_to_num(np.abs(distances_mapping[k1]) < sd_mapping[k1])
                    if not interact:
                        yield close
                    else:
                        for k2 in range(K):
                            yield close * np.nan_to_num(distances_mapping[k2])
                else:
                    raise ValueError("version must be 'local' or 'quadratic'.")

        # append instrument blocks
        other_blocks.append([])
        rival_blocks.append([])
        ownership = firm_ids[indices_t] == firm_ids[indices_t].T
        for term in generate_instrument_terms():
            other_blocks[-1].append((ownership * term).sum(axis=1, keepdims=True))
            rival_blocks[-1].append((~ownership * term).sum(axis=1, keepdims=True))

    # stack the blocks
    return np.c_[np.block(other_blocks), np.block(rival_blocks)]


def build_integration(integration: Integration, dimensions: int) -> RecArray:
    r"""Build nodes and weights for integration over agent choice probabilities.

    This function can be used to build custom ``agent_data`` for :class:`Problem` initialization. Specifically, this
    function affords more flexibility than passing an :class:`Integration` configuration directly to :class:`Problem`.
    For example, if agents have unobserved tastes over only a subset of demand-side nonlinear product characteristics
    (i.e., if ``sigma`` in :meth:`Problem.solve` has columns of zeros), this function can be used to build agent data
    with fewer columns of integration nodes than the number of unobserved product characteristics, :math:`K_2`. This
    function can also be used to construct nodes that can be transformed into demographic variables.

    To build nodes and weights for multiple markets, this function can be called multiple times, once for each market.

    Parameters
    ----------
    integration : `Integration`
        :class:`Integration` configuration for how to build nodes and weights for integration.
    dimensions : `int`
        Number of dimensions over which to integrate, or equivalently, the number of columns of integration nodes.
        When an :class:`Integration` configuration is passed directly to :class:`Problem`, this is the number of
        demand-side nonlinear product characteristics, :math:`K_2`.

    Returns
    -------
    `recarray`
        Nodes and weights for integration over agent utilities. Fields:

            - **weights** : (`numeric`) - Integration weights, :math:`w`.

            - **nodes** : (`numeric`) - Unobserved agent characteristics called integration nodes, :math:`\nu`.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_integration.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # validate the arguments
    if not isinstance(integration, Integration):
        raise TypeError("integration must be an Integration instance.")
    if not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("dimensions must be a positive integer.")

    # build and structure the nodes and weights
    nodes, weights = integration._build(dimensions)
    return structure_matrices({
        'weights': (weights, options.dtype),
        'nodes': (nodes, options.dtype)
    })


def build_matrix(formulation: Formulation, data: Mapping) -> Array:
    r"""Construct a matrix according to a formulation.

    Parameters
    ----------
    formulation : `Formulation`
        :class:`Formulation` configuration for the matrix. Variable names should correspond to fields in ``data``. The
        ``absorb`` argument of :class:`Formulation` can be used to absorb fixed effects after the matrix has been
        constructed.
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
    matrix = formulation._build_matrix(data)[0]
    if formulation._absorbed_terms:
        absorb = formulation._build_absorb(formulation._build_ids(data))
        matrix, errors = absorb(matrix)
        if errors:
            raise exceptions.MultipleErrors(errors)
    return matrix


def data_to_dict(data: RecArray) -> Dict[str, Array]:
    r"""Convert a NumPy record array into a dictionary.

    Most data in PyBLP are structured as NumPy record arrays (e.g., :attr:`Problem.products` and
    :attr:`SimulationResults.product_data`) which can be cumbersome to work with when working with data types that can't
    represent matrices, such as the :class:`pandas.DataFrame`.

    This function converts record arrays created by PyBLP into dictionaries that map field names to one-dimensional
    arrays. Matrices in the original record array (e.g., ``demand_instruments``) are split into as many fields as there
    are columns (e.g., ``demand_instruments0``, ``demand_instruments1``, and so on).

    Parameters
    ----------
    data : `recarray`
        Record array created by PyBLP.

    Returns
    -------
    `dict`
        The data re-structured as a dictionary.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/data_to_dict.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """
    if not isinstance(data, np.recarray):
        raise TypeError("data must be a NumPy record array.")
    mapping: Dict[str, Array] = {}
    for key in data.dtype.names:
        if len(data[key].shape) > 2:
            raise ValueError("Arrays with more than two dimensions are not supported.")
        if len(data[key].shape) == 1 or data[key].shape[1] == 1 or data[key].size == 0:
            mapping[key] = data[key].flatten()
            continue
        for index in range(data[key].shape[1]):
            new_key = f'{key}{index}'
            if new_key in data.dtype.names:
                raise KeyError(f"'{key}' cannot be split into columns because '{new_key}' is already a field.")
            mapping[new_key] = data[key][:, index].flatten()
    return mapping

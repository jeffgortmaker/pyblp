"""Construction of nodes and weights for integration."""

import functools
import itertools
from typing import Iterable, List, Optional, Tuple

import numpy as np
import scipy.special
import scipy.stats

from ..utilities.basics import Array, Options, StringRepresentation, format_options


class Integration(StringRepresentation):
    r"""Configuration for building integration nodes and weights.

    Parameters
    ----------
    specification : `str`
        How to build nodes and weights. One of the following:

            - ``'monte_carlo'`` - Draw from a pseudo-random standard multivariate normal distribution. Integration
              weights are ``1 / size``. The ``seed`` field of ``options`` can be used to seed the random number
              generator.

            - ``'halton'`` - Generate nodes according to the Halton. Different primes (2, 3, 5, etc.) are used for
              different dimensions. Integration weights are ``1 / size``. By default, the first ``100`` values in each
              dimension are discarded to eliminate correlation between dimensions. The ``discard`` field of ``options``
              can be used to increase this number.

            - ``'lhs'`` - Generate nodes according to Latin Hypercube Sampling (LHS). Integration weights are
              ``1 / size``. The ``seed`` field of ``options`` can be used to seed the random number generator.

            - ``'mlhs'`` - Generate nodes according to Modified Latin Hypercube Sampling (MLHS) described by
              :ref:`references:Hess, Train, and Polak (2004)`. Integration weights are ``1 / size``. The ``seed`` field
              of ``options`` can be used to seed the random number generator.

            - ``'product'`` - Generate nodes and weights according to the level-``size`` Gauss-Hermite product rule.

            - ``'nested_product'`` - Generate nodes and weights according to the level-``size`` nested Gauss-Hermite
              product rule. Weights can be negative.

            - ``'grid'`` - Generate a sparse grid of nodes and weights according to the level-``size`` Gauss-Hermite
              quadrature rule. Weights can be negative.

            - ``'nested_grid'`` - Generate a sparse grid of nodes and weights according to the level ``size`` nested
              Gauss-Hermite quadrature rule. Weights can be negative.

        Best practice for low dimensions is probably to use ``'product'`` to a relatively high degree of polynomial
        accuracy. In higher dimensions, ``'grid'`` appears to scale the best. For more information, see
        :ref:`references:Judd and Skrainka (2011)` and :ref:`references:Conlon and Gortmaker (2019)`.

        Sparse grids are constructed in analogously to the Matlab function `nwspgr <http://www.sparse-grids.de/>`_
        created by Florian Heiss and Viktor Winschel. For more information, see
        :ref:`references:Heiss and Winschel (2008)`.

    size : `int`
        The number of draws if ``specification`` is ``'monte_carlo'``, ``'lhs'``, or ``'mlhs'``, and the level of the
        quadrature rule otherwise.
    specification_options : `dict, optional`
        Options for the integration specification. The ``'monte_carlo'``, ``'lhs'``, and ``'mlhs'`` specifications
        support the following option:

            - **seed** : (`int`) - Passed to :class:`numpy.random.mtrand.RandomState` to seed the random number
              generator before building integration nodes. By default, a seed is not passed to the random number
              generator.

        The ``'halton'`` specification supports the following option:

            - **discard** : (`int`) - How many values at the beginning of each dimension's Halton sequence to discard.
              Discarding values at the start of each dimension's sequence is the simplest way to eliminate correlation
              between dimensions. By default, the first ``100`` values in each dimension are discarded.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/integration.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    _size: int
    _seed: Optional[int]
    _description: str
    _builder: functools.partial
    _specification_options: Options

    def __init__(self, specification: str, size: int, specification_options: Optional[Options] = None) -> None:
        """Validate the specification and identify the builder."""
        specifications = {
            'monte_carlo': (functools.partial(monte_carlo), "with Monte Carlo simulation"),
            'halton': (functools.partial(halton), "with Halton sequences"),
            'lhs': (functools.partial(lhs), "with Latin Hypercube Sampling (LHS)"),
            'mlhs': (functools.partial(lhs, modified=True), "with Modified Latin Hypercube Sampling (MLHS)"),
            'product': (functools.partial(product_rule), f"according to the level-{size} Gauss-Hermite product rule"),
            'grid': (
                functools.partial(sparse_grid),
                f"in a sparse grid according to the level-{size} Gauss-Hermite rule"
            ),
            'nested_product': (
                functools.partial(product_rule, nested=True),
                f"according to the level-{size} nested Gauss-Hermite product rule"
            ),
            'nested_grid': (
                functools.partial(sparse_grid, nested=True),
                f"in a sparse grid according to the level-{size} nested Gauss-Hermite rule"
            )
        }

        # validate the configuration
        if specification not in specifications:
            raise ValueError(f"specification must be one of {list(specifications.keys())}.")
        if not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer.")
        if specification_options is not None and not isinstance(specification_options, dict):
            raise ValueError("specification_options must be None or a dict.")

        # initialize class attributes
        self._size = size
        self._specification = specification
        self._builder, self._description = specifications[specification]

        # set default options
        self._specification_options: Options = {}
        if specification == 'halton':
            self._specification_options['discard'] = 100

        # update and validate options
        self._specification_options.update(specification_options or {})
        if specification in {'monte_carlo', 'lhs', 'mlhs'}:
            if not isinstance(self._specification_options.get('seed', 0), int):
                raise ValueError("The specification option seed must be an integer.")
        elif specification == 'halton':
            discard = self._specification_options['discard']
            if not isinstance(discard, int) or discard < 0:
                raise ValueError("The specification option discard must be a nonnegative integer.")

    def __str__(self) -> str:
        """Format the configuration as a string."""
        return (
            f"Configured to construct nodes and weights {self._description} with options "
            f"{format_options(self._specification_options)}."
        )

    def _build_many(self, dimensions: int, ids: Iterable) -> Tuple[Array, Array, Array]:
        """Build concatenated IDs, nodes, and weights for each ID."""
        builder = self._builder
        if self._specification in {'monte_carlo', 'lhs', 'mlhs'}:
            builder = functools.partial(builder, state=np.random.RandomState(self._specification_options.get('seed')))
        count = 0
        ids_list: List[Array] = []
        nodes_list: List[Array] = []
        weights_list: List[Array] = []
        for i in ids:
            if self._specification == 'halton':
                nodes, weights = builder(dimensions, self._size, start=self._specification_options['discard'] + count)
            else:
                nodes, weights = builder(dimensions, self._size)
            ids_list.append(np.repeat(i, weights.size))
            nodes_list.append(nodes)
            weights_list.append(weights)
            count += weights.size
        return np.concatenate(ids_list), np.concatenate(nodes_list), np.concatenate(weights_list)

    def _build(self, dimensions: int) -> Tuple[Array, Array]:
        """Build nodes and weights."""
        builder = self._builder
        if self._specification in {'monte_carlo', 'lhs', 'mlhs'}:
            builder = functools.partial(builder, state=np.random.RandomState(self._specification_options.get('seed')))
        if self._specification == 'halton':
            return builder(dimensions, self._size, start=self._specification_options['discard'])
        return builder(dimensions, self._size)


def monte_carlo(dimensions: int, size: int, state: np.random.RandomState) -> Tuple[Array, Array]:
    """Draw from a pseudo-random standard multivariate normal distribution."""
    nodes = state.normal(size=(size, dimensions))
    weights = np.repeat(1 / size, size)
    return nodes, weights


def halton(dimensions: int, size: int, start: int) -> Tuple[Array, Array]:
    """Generate nodes and weights for integration according to the Halton sequence."""

    # generate Halton sequences
    sequences = np.zeros((size, dimensions))
    for dimension in range(dimensions):
        base = get_prime(dimension)
        for index in range(size):
            value = 0.0
            denominator = 1.0
            quotient = start + index
            while quotient > 0:
                quotient, remainder = divmod(quotient, base)
                denominator *= base
                value += remainder / denominator
            sequences[index, dimension] = value

    # transform the sequences and construct weights
    nodes = scipy.stats.norm().ppf(sequences)
    weights = np.repeat(1 / size, size)
    return nodes, weights


def lhs(dimensions: int, size: int, state: np.random.RandomState, modified: bool = False) -> Tuple[Array, Array]:
    """Use Latin Hypercube Sampling to generate nodes and weights for integration."""

    # generate the samples
    samples = np.zeros((size, dimensions))
    for dimension in range(dimensions):
        samples[:, dimension] = state.permutation(np.arange(size) + state.uniform(size=1 if modified else size)) / size

    # transform the samples and construct weights
    nodes = scipy.stats.norm().ppf(samples)
    weights = np.repeat(1 / size, size)
    return nodes, weights


@functools.lru_cache()
def product_rule(dimensions: int, level: int, nested: bool = False) -> Tuple[Array, Array]:
    """Generate nodes and weights for integration according to the Gauss-Hermite product rule or its nested analog."""
    base_nodes, base_weights = quadrature_rule(level, nested)
    nodes = np.array(list(itertools.product(base_nodes, repeat=dimensions)))
    weights = functools.reduce(np.kron, itertools.repeat(base_weights, dimensions))
    return nodes, weights


@functools.lru_cache()
def sparse_grid(dimensions: int, level: int, nested: bool = False) -> Tuple[Array, Array]:
    """Generate a sparse grid of nodes and weights according to the univariate Gauss-Hermite quadrature rule or its
    nested analog.
    """

    # construct nodes and weights
    nodes_list: List[Array] = []
    weights_list: List[Array] = []
    for q in range(max(0, level - dimensions), level):
        # compute the combinatorial coefficient applied to the component product rules
        coefficient = (-1)**(level - q - 1) * scipy.special.binom(dimensions - 1, dimensions + q - level)

        # compute product rules for each level in all dimensions-length sequences that sum to dimensions + q
        for base_levels in same_size_sequences(dimensions, dimensions + q):
            base_nodes_list, base_weights_list = zip(*(quadrature_rule(l, nested) for l in base_levels))
            nodes_list.append(np.array(list(itertools.product(*base_nodes_list))))
            weights_list.append(coefficient * functools.reduce(np.kron, base_weights_list))

    # combine the lists of nodes and weights into arrays
    nodes = np.concatenate(nodes_list)
    weights = np.concatenate(weights_list)

    # sort nodes and weights by the first column of nodes, then by the second column, and so on
    sorted_indices = np.lexsort(nodes[:, ::-1].T)
    nodes = nodes[sorted_indices]
    weights = weights[sorted_indices]

    # merge weights for repeated rows, keeping only one set of nodes
    last = 0
    keep = [last]
    for row in range(1, weights.size):
        if np.array_equal(nodes[row], nodes[row - 1]):
            weights[last] += weights[row]
            continue
        last = row
        keep.append(row)
    nodes = nodes[keep]
    weights = weights[keep]

    # normalize the weights
    weights /= weights.sum()
    return nodes, weights


def same_size_sequences(size: int, summation: int) -> Array:
    """Compute all sequences of positive integers with a fixed size that sum to a fixed number. The algorithm was
    written to allow for with vectors that can take on zero, so we subtract the fixed size from the fixed summation at
    the beginning and then increment the sequences by one at the end.
    """
    summation -= size
    sequence = np.zeros(size, np.int64)
    sequence[0] = summation
    sequences = [sequence.copy()]
    forward = 0
    while sequence[-1] < summation:
        if forward == size - 1:
            for backward in reversed(range(forward)):
                forward = backward
                if sequence[backward] != 0:
                    break
        sequence[forward] -= 1
        forward += 1
        sequence[forward] = summation - sequence[:forward].sum()
        if forward < size - 1:
            sequence[forward + 1:] = 0
        sequences.append(sequence.copy())
    return np.vstack(sequences) + 1


def quadrature_rule(level: int, nested: bool) -> Tuple[Array, Array]:
    """Compute nodes and weights for the univariate Gauss-Hermite quadrature rule or its nested analog."""
    if not nested:
        raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(level)
        return raw_nodes * np.sqrt(2), raw_weights / np.sqrt(np.pi)
    node_data, weight_data = get_nested_data(level)
    return np.r_[-node_data[::-1], 0, node_data], np.r_[weight_data[::-1], weight_data[1:]]


def get_prime(dimension: int) -> int:
    """Return the prime number corresponding to a dimension when constructing a Halton sequence."""
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
        109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
        367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491,
        499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641,
        643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787,
        797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
        947, 953, 967, 971, 977, 983, 991, 997
    ]
    try:
        return primes[dimension]
    except IndexError:
        raise ValueError(f"Halton sequences are only available for {len(primes)} dimensions here.")


def get_nested_data(level: int) -> Tuple[Array, Array]:
    """Return node and weight data used to construct the nested Gauss-Hermite rule."""
    node_data_list = [
        [],
        [1.7320508075688772e+00],
        [1.7320508075688772e+00],
        [7.4109534999454085e-01, 1.7320508075688772e+00, 4.1849560176727323e+00],
        [7.4109534999454085e-01, 1.7320508075688772e+00, 2.8612795760570582e+00, 4.1849560176727323e+00],
        [7.4109534999454085e-01, 1.7320508075688772e+00, 2.8612795760570582e+00, 4.1849560176727323e+00],
        [7.4109534999454085e-01, 1.7320508075688772e+00, 2.8612795760570582e+00, 4.1849560176727323e+00],
        [7.4109534999454085e-01, 1.7320508075688772e+00, 2.8612795760570582e+00, 4.1849560176727323e+00],
        [
            7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00, 2.5960831150492023e+00,
            2.8612795760570582e+00, 4.1849560176727323e+00, 5.1870160399136562e+00, 6.3633944943363696e+00
        ],
        [
            7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00, 2.5960831150492023e+00,
            2.8612795760570582e+00, 3.2053337944991944e+00, 4.1849560176727323e+00, 5.1870160399136562e+00,
            6.3633944943363696e+00
        ],
        [
            7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00, 2.5960831150492023e+00,
            2.8612795760570582e+00, 3.2053337944991944e+00, 4.1849560176727323e+00, 5.1870160399136562e+00,
            6.3633944943363696e+00
        ],
        [
            7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00, 2.5960831150492023e+00,
            2.8612795760570582e+00, 3.2053337944991944e+00, 4.1849560176727323e+00, 5.1870160399136562e+00,
            6.3633944943363696e+00
        ],
        [
            7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00, 2.5960831150492023e+00,
            2.8612795760570582e+00, 3.2053337944991944e+00, 4.1849560176727323e+00, 5.1870160399136562e+00,
            6.3633944943363696e+00
        ],
        [
            7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00, 2.5960831150492023e+00,
            2.8612795760570582e+00, 3.2053337944991944e+00, 4.1849560176727323e+00, 5.1870160399136562e+00,
            6.3633944943363696e+00
        ],
        [
            7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00, 2.5960831150492023e+00,
            2.8612795760570582e+00, 3.2053337944991944e+00, 4.1849560176727323e+00, 5.1870160399136562e+00,
            6.3633944943363696e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 5.1870160399136562e+00, 6.3633944943363696e+00,
            7.1221067008046166e+00, 7.9807717985905606e+00, 9.0169397898903032e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 5.1870160399136562e+00, 5.6981777684881099e+00,
            6.3633944943363696e+00, 7.1221067008046166e+00, 7.9807717985905606e+00, 9.0169397898903032e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 4.7364330859522967e+00, 5.1870160399136562e+00,
            5.6981777684881099e+00, 6.3633944943363696e+00, 7.1221067008046166e+00, 7.9807717985905606e+00,
            9.0169397898903032e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 4.7364330859522967e+00, 5.1870160399136562e+00,
            5.6981777684881099e+00, 6.3633944943363696e+00, 7.1221067008046166e+00, 7.9807717985905606e+00,
            9.0169397898903032e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 4.7364330859522967e+00, 5.1870160399136562e+00,
            5.6981777684881099e+00, 6.3633944943363696e+00, 7.1221067008046166e+00, 7.9807717985905606e+00,
            9.0169397898903032e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 4.7364330859522967e+00, 5.1870160399136562e+00,
            5.6981777684881099e+00, 6.3633944943363696e+00, 7.1221067008046166e+00, 7.9807717985905606e+00,
            9.0169397898903032e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 4.7364330859522967e+00, 5.1870160399136562e+00,
            5.6981777684881099e+00, 6.3633944943363696e+00, 7.1221067008046166e+00, 7.9807717985905606e+00,
            9.0169397898903032e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 4.7364330859522967e+00, 5.1870160399136562e+00,
            5.6981777684881099e+00, 6.3633944943363696e+00, 7.1221067008046166e+00, 7.9807717985905606e+00,
            9.0169397898903032e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 4.7364330859522967e+00, 5.1870160399136562e+00,
            5.6981777684881099e+00, 6.3633944943363696e+00, 7.1221067008046166e+00, 7.9807717985905606e+00,
            9.0169397898903032e+00
        ],
        [
            2.4899229757996061e-01, 7.4109534999454085e-01, 1.2304236340273060e+00, 1.7320508075688772e+00,
            2.2336260616769419e+00, 2.5960831150492023e+00, 2.8612795760570582e+00, 3.2053337944991944e+00,
            3.6353185190372783e+00, 4.1849560176727323e+00, 4.7364330859522967e+00, 5.1870160399136562e+00,
            5.6981777684881099e+00, 6.3633944943363696e+00, 7.1221067008046166e+00, 7.9807717985905606e+00,
            9.0169397898903032e+00
        ]
    ]
    weight_data_list = [
        [+1.0000000000000000e+00],
        [+6.6666666666666663e-01, +1.6666666666666666e-01],
        [+6.6666666666666674e-01, +1.6666666666666666e-01],
        [+4.5874486825749189e-01, +1.3137860698313561e-01, +1.3855327472974924e-01, +6.9568415836913987e-04],
        [
            +2.5396825396825407e-01, +2.7007432957793776e-01, +9.4850948509485125e-02, +7.9963254708935293e-03,
            +9.4269457556517470e-05
        ],
        [
            +2.5396825396825429e-01, +2.7007432957793776e-01, +9.4850948509485070e-02, +7.9963254708935293e-03,
            +9.4269457556517551e-05
        ],
        [
            +2.5396825396825418e-01, +2.7007432957793781e-01, +9.4850948509485014e-02, +7.9963254708935311e-03,
            +9.4269457556517592e-05
        ],
        [
            +2.5396825396825418e-01, +2.7007432957793781e-01, +9.4850948509485042e-02, +7.9963254708935276e-03,
            +9.4269457556517375e-05
        ],
        [
            +2.6692223033505302e-01, +2.5456123204171222e-01, +1.4192654826449365e-02, +8.8681002152028010e-02,
            +1.9656770938777492e-03, +7.0334802378279075e-03, +1.0563783615416941e-04, -8.2049207541509217e-07,
            +2.1136499505424257e-08
        ],
        [
            +3.0346719985420623e-01, +2.0832499164960877e-01, +6.1151730125247716e-02, +6.4096054686807610e-02,
            +1.8085234254798462e-02, -6.3372247933737571e-03, +2.8848804365067559e-03, +6.0123369459847997e-05,
            +6.0948087314689840e-07, +8.6296846022298632e-10
        ],
        [
            +3.0346719985420623e-01, +2.0832499164960872e-01, +6.1151730125247709e-02, +6.4096054686807541e-02,
            +1.8085234254798459e-02, -6.3372247933737545e-03, +2.8848804365067555e-03, +6.0123369459847922e-05,
            +6.0948087314689830e-07, +8.6296846022298839e-10
        ],
        [
            +3.0346719985420623e-01, +2.0832499164960872e-01, +6.1151730125247716e-02, +6.4096054686807624e-02,
            +1.8085234254798466e-02, -6.3372247933737545e-03, +2.8848804365067559e-03, +6.0123369459847841e-05,
            +6.0948087314689830e-07, +8.6296846022298963e-10
        ],
        [
            +3.0346719985420600e-01, +2.0832499164960883e-01, +6.1151730125247730e-02, +6.4096054686807638e-02,
            +1.8085234254798459e-02, -6.3372247933737580e-03, +2.8848804365067555e-03, +6.0123369459847868e-05,
            +6.0948087314689830e-07, +8.6296846022298756e-10
        ],
        [
            +3.0346719985420617e-01, +2.0832499164960874e-01, +6.1151730125247702e-02, +6.4096054686807596e-02,
            +1.8085234254798459e-02, -6.3372247933737563e-03, +2.8848804365067555e-03, +6.0123369459847936e-05,
            +6.0948087314689851e-07, +8.6296846022298322e-10
        ],
        [
            +3.0346719985420612e-01, +2.0832499164960874e-01, +6.1151730125247723e-02, +6.4096054686807652e-02,
            +1.8085234254798459e-02, -6.3372247933737597e-03, +2.8848804365067563e-03, +6.0123369459848091e-05,
            +6.0948087314689851e-07, +8.6296846022298983e-10
        ],
        [
            +2.5890005324151566e-01, +2.8128101540033167e-02, +1.9968863511734550e-01, +6.5417392836092561e-02,
            +6.1718532565867179e-02, +1.7608475581318002e-03, +1.6592492698936010e-02, -5.5610063068358157e-03,
            +2.7298430467334002e-03, +1.5044205390914219e-05, +5.9474961163931621e-05, +6.1435843232617913e-07,
            +7.9298267864869338e-10, +5.1158053105504208e-12, -1.4840835740298868e-13, +1.2618464280815118e-15
        ],
        [
            +1.3911022236338039e-01, +1.0387687125574284e-01, +1.7607598741571459e-01, +7.7443602746299481e-02,
            +5.4677556143463042e-02, +7.3530110204955076e-03, +1.1529247065398790e-02, -2.7712189007789243e-03,
            +2.1202259559596325e-03, +8.3236045295766745e-05, +5.5691158981081479e-05, +6.9086261179113738e-07,
            -1.3486017348542930e-08, +1.5542195992782658e-09, -1.9341305000880955e-11, +2.6640625166231651e-13,
            -9.9313913286822465e-16
        ],
        [
            +5.1489450806921377e-04, +1.9176011588804434e-01, +1.4807083115521585e-01, +9.2364726716986353e-02,
            +4.5273685465150391e-02, +1.5673473751851151e-02, +3.1554462691875513e-03, +2.3113452403522071e-03,
            +8.1895392750226735e-04, +2.7524214116785131e-04, +3.5729348198975332e-05, +2.7342206801187888e-06,
            +2.4676421345798140e-07, +2.1394194479561062e-08, +4.6011760348655917e-10, +3.0972223576062995e-12,
            +5.4500412650638128e-15, +1.0541326582334014e-18
        ],
        [
            +5.1489450806921377e-04, +1.9176011588804437e-01, +1.4807083115521585e-01, +9.2364726716986353e-02,
            +4.5273685465150523e-02, +1.5673473751851151e-02, +3.1554462691875604e-03, +2.3113452403522050e-03,
            +8.1895392750226670e-04, +2.7524214116785131e-04, +3.5729348198975447e-05, +2.7342206801187884e-06,
            +2.4676421345798140e-07, +2.1394194479561056e-08, +4.6011760348656077e-10, +3.0972223576063011e-12,
            +5.4500412650637663e-15, +1.0541326582337958e-18
        ],
        [
            +5.1489450806925551e-04, +1.9176011588804440e-01, +1.4807083115521585e-01, +9.2364726716986298e-02,
            +4.5273685465150537e-02, +1.5673473751851155e-02, +3.1554462691875573e-03, +2.3113452403522080e-03,
            +8.1895392750226724e-04, +2.7524214116785137e-04, +3.5729348198975352e-05, +2.7342206801187888e-06,
            +2.4676421345798124e-07, +2.1394194479561056e-08, +4.6011760348656144e-10, +3.0972223576062963e-12,
            +5.4500412650638365e-15, +1.0541326582335402e-18
        ],
        [
            +5.1489450806913744e-04, +1.9176011588804429e-01, +1.4807083115521594e-01, +9.2364726716986312e-02,
            +4.5273685465150391e-02, +1.5673473751851151e-02, +3.1554462691875565e-03, +2.3113452403522089e-03,
            +8.1895392750226670e-04, +2.7524214116785142e-04, +3.5729348198975285e-05, +2.7342206801187888e-06,
            +2.4676421345798119e-07, +2.1394194479561059e-08, +4.6011760348656594e-10, +3.0972223576062950e-12,
            +5.4500412650638696e-15, +1.0541326582332041e-18
        ],
        [
            +5.1489450806903368e-04, +1.9176011588804448e-01, +1.4807083115521574e-01, +9.2364726716986423e-02,
            +4.5273685465150516e-02, +1.5673473751851161e-02, +3.1554462691875543e-03, +2.3113452403522063e-03,
            +8.1895392750226713e-04, +2.7524214116785164e-04, +3.5729348198975319e-05, +2.7342206801187905e-06,
            +2.4676421345798151e-07, +2.1394194479561082e-08, +4.6011760348656005e-10, +3.0972223576063043e-12,
            +5.4500412650637592e-15, +1.0541326582339926e-18
        ],
        [
            +5.1489450806913755e-04, +1.9176011588804442e-01, +1.4807083115521577e-01, +9.2364726716986381e-02,
            +4.5273685465150468e-02, +1.5673473751851155e-02, +3.1554462691875560e-03, +2.3113452403522045e-03,
            +8.1895392750226572e-04, +2.7524214116785158e-04, +3.5729348198975298e-05, +2.7342206801187892e-06,
            +2.4676421345798129e-07, +2.1394194479561072e-08, +4.6011760348656103e-10, +3.0972223576062963e-12,
            +5.4500412650638207e-15, +1.0541326582338368e-18
        ],
        [
            +5.1489450806914438e-04, +1.9176011588804442e-01, +1.4807083115521577e-01, +9.2364726716986340e-02,
            +4.5273685465150509e-02, +1.5673473751851155e-02, +3.1554462691875586e-03, +2.3113452403522058e-03,
            +8.1895392750226551e-04, +2.7524214116785142e-04, +3.5729348198975386e-05, +2.7342206801187884e-06,
            +2.4676421345798082e-07, +2.1394194479561059e-08, +4.6011760348656382e-10, +3.0972223576062942e-12,
            +5.4500412650638381e-15, +1.0541326582336941e-18
        ],
        [
            +5.1489450806919989e-04, +1.9176011588804437e-01, +1.4807083115521580e-01, +9.2364726716986395e-02,
            +4.5273685465150426e-02, +1.5673473751851158e-02, +3.1554462691875539e-03, +2.3113452403522054e-03,
            +8.1895392750226681e-04, +2.7524214116785142e-04, +3.5729348198975292e-05, +2.7342206801187884e-06,
            +2.4676421345798108e-07, +2.1394194479561056e-08, +4.6011760348655901e-10, +3.0972223576062975e-12,
            +5.4500412650638412e-15, +1.0541326582337527e-18
        ]
    ]
    try:
        node_data = np.array(node_data_list[level - 1])
        weight_data = np.array(weight_data_list[level - 1])
    except IndexError:
        raise ValueError(f"The nested rule is only available up to a level of {len(node_data_list)}.")
    return node_data, weight_data

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

            - ``'halton'`` - Generate nodes according to the Halton. A different prime (starting with 2, 3, 5, etc.) is
              used for each dimension of integration. To eliminate correlation between dimensions, the first ``1000``
              values are by default discarded in each dimension. To further improve performance (particularly in
              settings with many dimensions), sequences are also by default scrambled with the algorithm of
              :ref:`references:Owen (2017)`. The ``discard``, ``scramble``, and ``seed`` fields of ``options`` can be
              used to configure these default settings.

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
        accuracy. In higher dimensions, ``'grid'`` or ``'halton'`` appears to scale the best. For more information, see
        :ref:`references:Judd and Skrainka (2011)` and :ref:`references:Conlon and Gortmaker (2020)`.

        Sparse grids are constructed in analogously to the Matlab function `nwspgr <http://www.sparse-grids.de/>`_
        created by Florian Heiss and Viktor Winschel. For more information, see
        :ref:`references:Heiss and Winschel (2008)`.

    size : `int`
        The number of draws if ``specification`` is ``'monte_carlo'``, ``'halton'``, ``'lhs'``, or ``'mlhs'``, and the
        level of the quadrature rule otherwise.
    specification_options : `dict, optional`
        Options for the integration specification. The ``'monte_carlo'``, ``'halton'``, ``'lhs'``, and ``'mlhs'``
        specifications support the following option:

            - **seed** : (`int`) - Passed to :class:`numpy.random.RandomState` to seed the random number
              generator before building integration nodes. By default, a seed is not passed to the random number
              generator. For ``'halton'`` draws, this is only relevant if ``scramble`` is ``True`` (which is the
              default).

        The ``'halton'`` specification supports the following options:

            - **discard** : (`int`) - How many values at the beginning of each dimension's Halton sequence to discard.
              Discarding values at the start of each dimension's sequence is the simplest way to eliminate correlation
              between dimensions. By default, the first ``1000`` values in each dimension are discarded.

            - **scramble** : (`bool`) - Whether to scramble the sequences with the algorithm of
              :ref:`references:Owen (2017)`. By default, sequences are scrambled.

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
            self._specification_options.update({
                'discard': 1000,
                'scramble': True,
            })

        # update and validate options
        self._specification_options.update(specification_options or {})
        if specification in {'monte_carlo', 'halton', 'lhs', 'mlhs'}:
            if not isinstance(self._specification_options.get('seed', 0), int):
                raise ValueError("The specification option seed must be an integer.")
        if specification == 'halton':
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

        # seed any underlying random number generator
        builder = self._builder
        if self._specification in {'monte_carlo', 'halton', 'lhs', 'mlhs'}:
            builder = functools.partial(builder, state=np.random.RandomState(self._specification_options.get('seed')))

        # build the arrays of IDs, nodes, and weights ID-by-ID
        count = 0
        ids_list: List[Array] = []
        nodes_list: List[Array] = []
        weights_list: List[Array] = []
        for i in ids:
            if self._specification == 'halton':
                start = self._specification_options['discard'] + count
                nodes, weights = builder(dimensions, self._size, start, self._specification_options['scramble'])
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
        if self._specification in {'monte_carlo', 'halton', 'lhs', 'mlhs'}:
            builder = functools.partial(builder, state=np.random.RandomState(self._specification_options.get('seed')))
        if self._specification == 'halton':
            start = self._specification_options['discard']
            return builder(dimensions, self._size, start, self._specification_options['scramble'])
        return builder(dimensions, self._size)


def monte_carlo(dimensions: int, size: int, state: np.random.RandomState) -> Tuple[Array, Array]:
    """Draw from a pseudo-random standard multivariate normal distribution."""
    nodes = state.normal(size=(size, dimensions))
    weights = np.repeat(1 / size, size)
    return nodes, weights


def halton(dimensions: int, size: int, start: int, scramble: bool, state: np.random.RandomState) -> Tuple[Array, Array]:
    """Generate nodes and weights for integration according to the Halton sequence."""

    # generate Halton sequences
    sequences = np.zeros((size, dimensions))
    for dimension in range(dimensions):
        base = get_prime(dimension)
        factor = 1 / base
        indices = np.arange(start, start + size)
        while 1 - factor < 1:
            indices, remainders = np.divmod(indices, base)
            if scramble:
                remainders = state.permutation(base)[remainders]
            sequences[:, dimension] += factor * remainders
            factor /= base

    # transform the sequences and construct weights
    nodes = scipy.stats.norm().ppf(sequences)
    weights = np.repeat(1 / size, size)
    return nodes, weights


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
    node_data, weight_data = get_quadrature_data(level, nested)
    base_nodes = np.r_[-node_data[::-1], node_data[1 if nested else level % 2:]]
    base_weights = np.r_[weight_data[::-1], weight_data[1 if nested else level % 2:]]
    nodes = np.array(list(itertools.product(base_nodes, repeat=dimensions)))
    weights = functools.reduce(np.kron, itertools.repeat(base_weights, dimensions))
    return nodes, weights


@functools.lru_cache()
def sparse_grid(dimensions: int, level: int, nested: bool = False) -> Tuple[Array, Array]:
    """Generate a sparse grid of nodes and weights according to the univariate Gauss-Hermite quadrature rule or its
    nested analog.
    """
    nodes = np.zeros((0, dimensions), np.float64)
    weights = np.zeros(0, np.float64)
    for q in range(max(0, level - dimensions), level):
        # compute the combinatorial coefficient applied to the component product rules
        coefficient = (-1)**(level - q - 1) * scipy.special.binom(dimensions - 1, dimensions + q - level)

        # compute product rules for each level in all dimensions-length sequences that sum to dimensions + q
        for base_levels in same_size_sequences(dimensions, dimensions + q):
            base_nodes_list, base_weights_list = zip(*(get_quadrature_data(l, nested) for l in base_levels))
            nodes = np.r_[nodes, np.array(list(itertools.product(*base_nodes_list)))]
            weights = np.r_[weights, coefficient * functools.reduce(np.kron, base_weights_list)]

        # sort nodes and weights by the first column of nodes, then by the second column, and so on
        sorted_indices = np.lexsort(nodes[:, ::-1].T)
        nodes = nodes[sorted_indices]
        weights = weights[sorted_indices]

        # merge weights for repeated rows
        last = 0
        keep = [last]
        for row in range(1, weights.size):
            if np.array_equal(nodes[row], nodes[row - 1]):
                weights[last] += weights[row]
            else:
                last = row
                keep.append(row)

        # keep only one set of nodes
        nodes = nodes[keep]
        weights = weights[keep]

    # copy to the other orthants
    middle_node = get_quadrature_data(1, nested)[0][0]
    for dimension in range(dimensions):
        copy = []
        for row in range(weights.size):
            if nodes[row, dimension] != middle_node:
                copy.append(row)
        if copy:
            nodes = np.r_[nodes, nodes[copy]]
            weights = np.r_[weights, weights[copy]]
            nodes[:-len(copy), dimension] = 2 * middle_node - nodes[:-len(copy), dimension]

    # re-sort and normalize weights
    sorted_indices = np.lexsort(nodes[:, ::-1].T)
    nodes = nodes[sorted_indices]
    weights = weights[sorted_indices]
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


def get_quadrature_data(level: int, nested: bool) -> Tuple[Array, Array]:
    """Compute nodes and weights for the univariate Gauss-Hermite quadrature rule or its nested analog."""
    if nested:
        node_data_list = [
            [0.0000000000000000e+000],
            [0.0000000000000000e+000, 1.7320508075688772e+000],
            [0.0000000000000000e+000, 1.7320508075688772e+000],
            [0.0000000000000000e+000, 7.4109534999454085e-001, 1.7320508075688772e+000, 4.1849560176727323e+000],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.7320508075688772e+000, 2.8612795760570582e+000,
                4.1849560176727323e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.7320508075688772e+000, 2.8612795760570582e+000,
                4.1849560176727323e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.7320508075688772e+000, 2.8612795760570582e+000,
                4.1849560176727323e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.7320508075688772e+000, 2.8612795760570582e+000,
                4.1849560176727323e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.2304236340273060e+000, 1.7320508075688772e+000,
                2.5960831150492023e+000, 2.8612795760570582e+000, 4.1849560176727323e+000, 5.1870160399136562e+000,
                6.3633944943363696e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.2304236340273060e+000, 1.7320508075688772e+000,
                2.5960831150492023e+000, 2.8612795760570582e+000, 3.2053337944991944e+000, 4.1849560176727323e+000,
                5.1870160399136562e+000, 6.3633944943363696e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.2304236340273060e+000, 1.7320508075688772e+000,
                2.5960831150492023e+000, 2.8612795760570582e+000, 3.2053337944991944e+000, 4.1849560176727323e+000,
                5.1870160399136562e+000, 6.3633944943363696e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.2304236340273060e+000, 1.7320508075688772e+000,
                2.5960831150492023e+000, 2.8612795760570582e+000, 3.2053337944991944e+000, 4.1849560176727323e+000,
                5.1870160399136562e+000, 6.3633944943363696e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.2304236340273060e+000, 1.7320508075688772e+000,
                2.5960831150492023e+000, 2.8612795760570582e+000, 3.2053337944991944e+000, 4.1849560176727323e+000,
                5.1870160399136562e+000, 6.3633944943363696e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.2304236340273060e+000, 1.7320508075688772e+000,
                2.5960831150492023e+000, 2.8612795760570582e+000, 3.2053337944991944e+000, 4.1849560176727323e+000,
                5.1870160399136562e+000, 6.3633944943363696e+000
            ],
            [
                0.0000000000000000e+000, 7.4109534999454085e-001, 1.2304236340273060e+000, 1.7320508075688772e+000,
                2.5960831150492023e+000, 2.8612795760570582e+000, 3.2053337944991944e+000, 4.1849560176727323e+000,
                5.1870160399136562e+000, 6.3633944943363696e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 5.1870160399136562e+000,
                6.3633944943363696e+000, 7.1221067008046166e+000, 7.9807717985905606e+000, 9.0169397898903032e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 5.1870160399136562e+000,
                5.6981777684881099e+000, 6.3633944943363696e+000, 7.1221067008046166e+000, 7.9807717985905606e+000,
                9.0169397898903032e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 4.7364330859522967e+000,
                5.1870160399136562e+000, 5.6981777684881099e+000, 6.3633944943363696e+000, 7.1221067008046166e+000,
                7.9807717985905606e+000, 9.0169397898903032e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 4.7364330859522967e+000,
                5.1870160399136562e+000, 5.6981777684881099e+000, 6.3633944943363696e+000, 7.1221067008046166e+000,
                7.9807717985905606e+000, 9.0169397898903032e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 4.7364330859522967e+000,
                5.1870160399136562e+000, 5.6981777684881099e+000, 6.3633944943363696e+000, 7.1221067008046166e+000,
                7.9807717985905606e+000, 9.0169397898903032e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 4.7364330859522967e+000,
                5.1870160399136562e+000, 5.6981777684881099e+000, 6.3633944943363696e+000, 7.1221067008046166e+000,
                7.9807717985905606e+000, 9.0169397898903032e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 4.7364330859522967e+000,
                5.1870160399136562e+000, 5.6981777684881099e+000, 6.3633944943363696e+000, 7.1221067008046166e+000,
                7.9807717985905606e+000, 9.0169397898903032e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 4.7364330859522967e+000,
                5.1870160399136562e+000, 5.6981777684881099e+000, 6.3633944943363696e+000, 7.1221067008046166e+000,
                7.9807717985905606e+000, 9.0169397898903032e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 4.7364330859522967e+000,
                5.1870160399136562e+000, 5.6981777684881099e+000, 6.3633944943363696e+000, 7.1221067008046166e+000,
                7.9807717985905606e+000, 9.0169397898903032e+000
            ],
            [
                0.0000000000000000e+000, 2.4899229757996061e-001, 7.4109534999454085e-001, 1.2304236340273060e+000,
                1.7320508075688772e+000, 2.2336260616769419e+000, 2.5960831150492023e+000, 2.8612795760570582e+000,
                3.2053337944991944e+000, 3.6353185190372783e+000, 4.1849560176727323e+000, 4.7364330859522967e+000,
                5.1870160399136562e+000, 5.6981777684881099e+000, 6.3633944943363696e+000, 7.1221067008046166e+000,
                7.9807717985905606e+000, 9.0169397898903032e+000
            ],
        ]
        weight_data_list = [
            [1.0000000000000000e+000],
            [6.6666666666666663e-001, 1.6666666666666666e-001],
            [6.6666666666666674e-001, 1.6666666666666666e-001],
            [4.5874486825749189e-001, 1.3137860698313561e-001, 1.3855327472974924e-001, 6.9568415836913987e-004],
            [
                2.5396825396825407e-001, 2.7007432957793776e-001, 9.4850948509485125e-002, 7.9963254708935293e-003,
                9.4269457556517470e-005],
            [
                2.5396825396825429e-001, 2.7007432957793776e-001, 9.4850948509485070e-002, 7.9963254708935293e-003,
                9.4269457556517551e-005],
            [
                2.5396825396825418e-001, 2.7007432957793781e-001, 9.4850948509485014e-002, 7.9963254708935311e-003,
                9.4269457556517592e-005],
            [
                2.5396825396825418e-001, 2.7007432957793781e-001, 9.4850948509485042e-002, 7.9963254708935276e-003,
                9.4269457556517375e-005],
            [
                2.6692223033505302e-001, 2.5456123204171222e-001, 1.4192654826449365e-002, 8.8681002152028010e-002,
                1.9656770938777492e-003, 7.0334802378279075e-003, 1.0563783615416941e-004, -8.2049207541509217e-007,
                2.1136499505424257e-008],
            [
                3.0346719985420623e-001, 2.0832499164960877e-001, 6.1151730125247716e-002, 6.4096054686807610e-002,
                1.8085234254798462e-002, -6.3372247933737571e-003, 2.8848804365067559e-003, 6.0123369459847997e-005,
                6.0948087314689840e-007, 8.6296846022298632e-010
            ],
            [
                3.0346719985420623e-001, 2.0832499164960872e-001, 6.1151730125247709e-002, 6.4096054686807541e-002,
                1.8085234254798459e-002, -6.3372247933737545e-003, 2.8848804365067555e-003, 6.0123369459847922e-005,
                6.0948087314689830e-007, 8.6296846022298839e-010
            ],
            [
                3.0346719985420623e-001, 2.0832499164960872e-001, 6.1151730125247716e-002, 6.4096054686807624e-002,
                1.8085234254798466e-002, -6.3372247933737545e-003, 2.8848804365067559e-003, 6.0123369459847841e-005,
                6.0948087314689830e-007, 8.6296846022298963e-010
            ],
            [
                3.0346719985420600e-001, 2.0832499164960883e-001, 6.1151730125247730e-002, 6.4096054686807638e-002,
                1.8085234254798459e-002, -6.3372247933737580e-003, 2.8848804365067555e-003, 6.0123369459847868e-005,
                6.0948087314689830e-007, 8.6296846022298756e-010
            ],
            [
                3.0346719985420617e-001, 2.0832499164960874e-001, 6.1151730125247702e-002, 6.4096054686807596e-002,
                1.8085234254798459e-002, -6.3372247933737563e-003, 2.8848804365067555e-003, 6.0123369459847936e-005,
                6.0948087314689851e-007, 8.6296846022298322e-010
            ],
            [
                3.0346719985420612e-001, 2.0832499164960874e-001, 6.1151730125247723e-002, 6.4096054686807652e-002,
                1.8085234254798459e-002, -6.3372247933737597e-003, 2.8848804365067563e-003, 6.0123369459848091e-005,
                6.0948087314689851e-007, 8.6296846022298983e-010
            ],
            [
                2.5890005324151566e-001, 2.8128101540033167e-002, 1.9968863511734550e-001, 6.5417392836092561e-002,
                6.1718532565867179e-002, 1.7608475581318002e-003, 1.6592492698936010e-002, -5.5610063068358157e-003,
                2.7298430467334002e-003, 1.5044205390914219e-005, 5.9474961163931621e-005, 6.1435843232617913e-007,
                7.9298267864869338e-010, 5.1158053105504208e-012, -1.4840835740298868e-013, 1.2618464280815118e-015
            ],
            [
                1.3911022236338039e-001, 1.0387687125574284e-001, 1.7607598741571459e-001, 7.7443602746299481e-002,
                5.4677556143463042e-002, 7.3530110204955076e-003, 1.1529247065398790e-002, -2.7712189007789243e-003,
                2.1202259559596325e-003, 8.3236045295766745e-005, 5.5691158981081479e-005, 6.9086261179113738e-007,
                -1.3486017348542930e-008, 1.5542195992782658e-009, -1.9341305000880955e-011, 2.6640625166231651e-013,
                -9.9313913286822465e-016
            ],
            [
                5.1489450806921377e-004, 1.9176011588804434e-001, 1.4807083115521585e-001, 9.2364726716986353e-002,
                4.5273685465150391e-002, 1.5673473751851151e-002, 3.1554462691875513e-003, 2.3113452403522071e-003,
                8.1895392750226735e-004, 2.7524214116785131e-004, 3.5729348198975332e-005, 2.7342206801187888e-006,
                2.4676421345798140e-007, 2.1394194479561062e-008, 4.6011760348655917e-010, 3.0972223576062995e-012,
                5.4500412650638128e-015, 1.0541326582334014e-018
            ],
            [
                5.1489450806921377e-004, 1.9176011588804437e-001, 1.4807083115521585e-001, 9.2364726716986353e-002,
                4.5273685465150523e-002, 1.5673473751851151e-002, 3.1554462691875604e-003, 2.3113452403522050e-003,
                8.1895392750226670e-004, 2.7524214116785131e-004, 3.5729348198975447e-005, 2.7342206801187884e-006,
                2.4676421345798140e-007, 2.1394194479561056e-008, 4.6011760348656077e-010, 3.0972223576063011e-012,
                5.4500412650637663e-015, 1.0541326582337958e-018
            ],
            [
                5.1489450806925551e-004, 1.9176011588804440e-001, 1.4807083115521585e-001, 9.2364726716986298e-002,
                4.5273685465150537e-002, 1.5673473751851155e-002, 3.1554462691875573e-003, 2.3113452403522080e-003,
                8.1895392750226724e-004, 2.7524214116785137e-004, 3.5729348198975352e-005, 2.7342206801187888e-006,
                2.4676421345798124e-007, 2.1394194479561056e-008, 4.6011760348656144e-010, 3.0972223576062963e-012,
                5.4500412650638365e-015, 1.0541326582335402e-018
            ],
            [
                5.1489450806913744e-004, 1.9176011588804429e-001, 1.4807083115521594e-001, 9.2364726716986312e-002,
                4.5273685465150391e-002, 1.5673473751851151e-002, 3.1554462691875565e-003, 2.3113452403522089e-003,
                8.1895392750226670e-004, 2.7524214116785142e-004, 3.5729348198975285e-005, 2.7342206801187888e-006,
                2.4676421345798119e-007, 2.1394194479561059e-008, 4.6011760348656594e-010, 3.0972223576062950e-012,
                5.4500412650638696e-015, 1.0541326582332041e-018
            ],
            [
                5.1489450806903368e-004, 1.9176011588804448e-001, 1.4807083115521574e-001, 9.2364726716986423e-002,
                4.5273685465150516e-002, 1.5673473751851161e-002, 3.1554462691875543e-003, 2.3113452403522063e-003,
                8.1895392750226713e-004, 2.7524214116785164e-004, 3.5729348198975319e-005, 2.7342206801187905e-006,
                2.4676421345798151e-007, 2.1394194479561082e-008, 4.6011760348656005e-010, 3.0972223576063043e-012,
                5.4500412650637592e-015, 1.0541326582339926e-018
            ],
            [
                5.1489450806913755e-004, 1.9176011588804442e-001, 1.4807083115521577e-001, 9.2364726716986381e-002,
                4.5273685465150468e-002, 1.5673473751851155e-002, 3.1554462691875560e-003, 2.3113452403522045e-003,
                8.1895392750226572e-004, 2.7524214116785158e-004, 3.5729348198975298e-005, 2.7342206801187892e-006,
                2.4676421345798129e-007, 2.1394194479561072e-008, 4.6011760348656103e-010, 3.0972223576062963e-012,
                5.4500412650638207e-015, 1.0541326582338368e-018
            ],
            [
                5.1489450806914438e-004, 1.9176011588804442e-001, 1.4807083115521577e-001, 9.2364726716986340e-002,
                4.5273685465150509e-002, 1.5673473751851155e-002, 3.1554462691875586e-003, 2.3113452403522058e-003,
                8.1895392750226551e-004, 2.7524214116785142e-004, 3.5729348198975386e-005, 2.7342206801187884e-006,
                2.4676421345798082e-007, 2.1394194479561059e-008, 4.6011760348656382e-010, 3.0972223576062942e-012,
                5.4500412650638381e-015, 1.0541326582336941e-018
            ],
            [
                5.1489450806919989e-004, 1.9176011588804437e-001, 1.4807083115521580e-001, 9.2364726716986395e-002,
                4.5273685465150426e-002, 1.5673473751851158e-002, 3.1554462691875539e-003, 2.3113452403522054e-003,
                8.1895392750226681e-004, 2.7524214116785142e-004, 3.5729348198975292e-005, 2.7342206801187884e-006,
                2.4676421345798108e-007, 2.1394194479561056e-008, 4.6011760348655901e-010, 3.0972223576062975e-012,
                5.4500412650638412e-015, 1.0541326582337527e-018
            ],
        ]
    else:
        node_data_list = [
            [0.0000000000000000e+000],
            [1.0000000000000002e+000],
            [0.0000000000000000e+000, 1.7320508075688772e+000],
            [7.4196378430272591e-001, 2.3344142183389773e+000],
            [0.0000000000000000e+000, 1.3556261799742659e+000, 2.8569700138728056e+000],
            [6.1670659019259422e-001, 1.8891758777537109e+000, 3.3242574335521193e+000],
            [0.0000000000000000e+000, 1.1544053947399682e+000, 2.3667594107345415e+000, 3.7504397177257425e+000],
            [5.3907981135137517e-001, 1.6365190424351082e+000, 2.8024858612875416e+000, 4.1445471861258945e+000],
            [
                0.0000000000000000e+000, 1.0232556637891326e+000, 2.0768479786778302e+000, 3.2054290028564703e+000,
                4.5127458633997826e+000
            ],
            [
                4.8493570751549764e-001, 1.4659890943911582e+000, 2.4843258416389546e+000, 3.5818234835519269e+000,
                4.8594628283323127e+000
            ],
            [
                0.0000000000000000e+000, 9.2886899738106388e-001, 1.8760350201548459e+000, 2.8651231606436447e+000,
                3.9361666071299775e+000, 5.1880012243748714e+000
            ],
            [
                4.4440300194413901e-001, 1.3403751971516167e+000, 2.2594644510007993e+000, 3.2237098287700974e+000,
                4.2718258479322815e+000, 5.5009017044677480e+000
            ],
            [
                0.0000000000000000e+000, 8.5667949351945005e-001, 1.7254183795882394e+000, 2.6206899734322149e+000,
                3.5634443802816347e+000, 4.5913984489365207e+000, 5.8001672523865011e+000
            ],
            [
                4.1259045795460181e-001, 1.2426889554854643e+000, 2.0883447457019444e+000, 2.9630365798386675e+000,
                3.8869245750597696e+000, 4.8969363973455646e+000, 6.0874095469012914e+000
            ],
            [
                0.0000000000000000e+000, 7.9912906832454811e-001, 1.6067100690287301e+000, 2.4324368270097581e+000,
                3.2890824243987664e+000, 4.1962077112690155e+000, 5.1900935913047821e+000, 6.3639478888298378e+000
            ],
            [
                3.8676060450055738e-001, 1.1638291005549648e+000, 1.9519803457163336e+000, 2.7602450476307019e+000,
                3.6008736241715487e+000, 4.4929553025200120e+000, 5.4722257059493433e+000, 6.6308781983931295e+000
            ],
            [
                0.0000000000000000e+000, 7.5184260070389630e-001, 1.5098833077967408e+000, 2.2810194402529889e+000,
                3.0737971753281941e+000, 3.9000657171980104e+000, 4.7785315896299840e+000, 5.7444600786594071e+000,
                6.8891224398953330e+000
            ],
            [
                3.6524575550769767e-001, 1.0983955180915013e+000, 1.8397799215086457e+000, 2.5958336889112403e+000,
                3.3747365357780907e+000, 4.1880202316294044e+000, 5.0540726854427405e+000, 6.0077459113595975e+000,
                7.1394648491464796e+000
            ],
            [
                0.0000000000000000e+000, 7.1208504404237993e-001, 1.4288766760783731e+000, 2.1555027613169351e+000,
                2.8980512765157536e+000, 3.6644165474506383e+000, 4.4658726268310316e+000, 5.3205363773360386e+000,
                6.2628911565132519e+000, 7.3825790240304316e+000
            ],
            [
                3.4696415708135592e-001, 1.0429453488027509e+000, 1.7452473208141270e+000, 2.4586636111723679e+000,
                3.1890148165533900e+000, 3.9439673506573163e+000, 4.7345813340460552e+000, 5.5787388058932015e+000,
                6.5105901570136551e+000, 7.6190485416797591e+000
            ],
            [
                0.0000000000000000e+000, 6.7804569244064405e-001, 1.3597658232112304e+000, 2.0491024682571628e+000,
                2.7505929810523733e+000, 3.4698466904753764e+000, 4.2143439816884216e+000, 4.9949639447820253e+000,
                5.8293820073044706e+000, 6.7514447187174609e+000, 7.8493828951138225e+000
            ],
            [
                3.3117931571527381e-001, 9.9516242227121554e-001, 1.6641248391179071e+000, 2.3417599962877080e+000,
                3.0324042278316763e+000, 3.7414963502665177e+000, 4.4763619773108685e+000, 5.2477244337144251e+000,
                6.0730749511228979e+000, 6.9859804240188152e+000, 8.0740299840217116e+000
            ],
            [
                0.0000000000000000e+000, 6.4847115353449580e-001, 1.2998764683039790e+000, 1.9573275529334242e+000,
                2.6243236340591820e+000, 3.3050400217529652e+000, 4.0047753217333044e+000, 4.7307241974514733e+000,
                5.4934739864717947e+000, 6.3103498544483996e+000, 7.2146594350518622e+000, 8.2933860274173536e+000
            ],
            [
                3.1737009662945231e-001, 9.5342192293210926e-001, 1.5934804298164202e+000, 2.2404678516917524e+000,
                2.8977286432233140e+000, 3.5693067640735610e+000, 4.2603836050199053e+000, 4.9780413746391208e+000,
                5.7327471752512009e+000, 6.5416750050986341e+000, 7.4378906660216630e+000, 8.5078035191952583e+000
            ],
            [
                0.0000000000000000e+000, 6.2246227918607611e-001, 1.2473119756167892e+000, 1.8770583699478387e+000,
                2.5144733039522058e+000, 3.1627756793881927e+000, 3.8259005699724917e+000, 4.5089299229672850e+000,
                5.2188480936442794e+000, 5.9660146906067020e+000, 6.7674649638097168e+000, 7.6560379553930762e+000,
                8.7175976783995885e+000
            ]
        ]
        weight_data_list = [
            [1.0000000000000000e+000],
            [5.0000000000000000e-001],
            [6.6666666666666663e-001, 1.6666666666666674e-001],
            [4.5412414523193145e-001, 4.5875854768068498e-002],
            [5.3333333333333344e-001, 2.2207592200561263e-001, 1.1257411327720691e-002],
            [4.0882846955602919e-001, 8.8615746041914523e-002, 2.5557844020562431e-003],
            [4.5714285714285757e-001, 2.4012317860501250e-001, 3.0757123967586491e-002, 5.4826885597221875e-004],
            [3.7301225767907736e-001, 1.1723990766175897e-001, 9.6352201207882630e-003, 1.1261453837536784e-004],
            [
                4.0634920634920685e-001, 2.4409750289493909e-001, 4.9916406765217969e-002, 2.7891413212317675e-003,
                2.2345844007746563e-005
            ],
            [
                3.4464233493201940e-001, 1.3548370298026730e-001, 1.9111580500770317e-002, 7.5807093431221972e-004,
                4.3106526307183106e-006
            ],
            [
                3.6940836940836957e-001, 2.4224029987397003e-001, 6.6138746071057644e-002, 6.7202852355372697e-003,
                1.9567193027122324e-004, 8.1218497902149036e-007
            ],
            [
                3.2166436151283007e-001, 1.4696704804532995e-001, 2.9116687912364138e-002, 2.2033806875331849e-003,
                4.8371849225906076e-005, 1.4999271676371597e-007
            ],
            [
                3.4099234099234149e-001, 2.3787152296413588e-001, 7.9168955860450141e-002, 1.1770560505996543e-002,
                6.8123635044292619e-004, 1.1526596527333885e-005, 2.7226276428059039e-008
            ],
            [
                3.0263462681301945e-001, 1.5408333984251366e-001, 3.8650108824253432e-002, 4.4289191069474066e-003,
                2.0033955376074381e-004, 2.6609913440676334e-006, 4.8681612577483872e-009
            ],
            [
                3.1825951825951820e-001, 2.3246229360973222e-001, 8.9417795399844444e-002, 1.7365774492137616e-002,
                1.5673575035499571e-003, 5.6421464051890157e-005, 5.9754195979205961e-007, 8.5896498996331805e-010
            ],
            [
                2.8656852123801241e-001, 1.5833837275094925e-001, 4.7284752354014067e-002, 7.2669376011847411e-003,
                5.2598492657390979e-004, 1.5300032162487286e-005, 1.3094732162868203e-007, 1.4978147231618314e-010
            ],
            [
                2.9953837012660756e-001, 2.2670630846897877e-001, 9.7406371162718081e-002, 2.3086657025711152e-002,
                2.8589460622846499e-003, 1.6849143155133945e-004, 4.0126794479798725e-006, 2.8080161179305783e-008,
                2.5843149193749151e-011
            ],
            [
                2.7278323465428789e-001, 1.6068530389351263e-001, 5.4896632480222654e-002, 1.0516517751941352e-002,
                1.0654847962916496e-003, 5.1798961441161962e-005, 1.0215523976369816e-006, 5.9054884788365484e-009,
                4.4165887693587078e-012
            ],
            [
                2.8377319275152108e-001, 2.2094171219914366e-001, 1.0360365727614400e-001, 2.8666691030118496e-002,
                4.5072354203420355e-003, 3.7850210941426759e-004, 1.5351145954666744e-005, 2.5322200320928681e-007,
                1.2203708484474786e-009, 7.4828300540572308e-013
            ],
            [
                2.6079306344955544e-001, 1.6173933398400026e-001, 6.1506372063976029e-002, 1.3997837447101043e-002,
                1.8301031310804918e-003, 1.2882627996192898e-004, 4.4021210902308646e-006, 6.1274902599829597e-008,
                2.4820623623151838e-010, 1.2578006724379305e-013
            ],
            [
                2.7026018357287707e-001, 2.1533371569505982e-001, 1.0839228562641938e-001, 3.3952729786542839e-002,
                6.4396970514087768e-003, 7.0804779548153736e-004, 4.2192347425515866e-005, 1.2253548361482522e-006,
                1.4506612844930740e-008, 4.9753686041217464e-011, 2.0989912195656652e-014
            ],
            [
                2.5024359658693501e-001, 1.6190629341367538e-001, 6.7196311428889891e-002, 1.7569072880805774e-002,
                2.8087610475772107e-003, 2.6228330325596416e-004, 1.3345977126808712e-005, 3.3198537498140043e-007,
                3.3665141594582109e-009, 9.8413789823460105e-012, 3.4794606478771428e-015
            ],
            [
                2.5850974080883904e-001, 2.0995966957754261e-001, 1.1207338260262091e-001, 3.8867183703480947e-002,
                8.5796783914656640e-003, 1.1676286374978613e-003, 9.3408186090312983e-005, 4.0899772449921549e-006,
                8.7750624838617161e-008, 7.6708888623999076e-010, 1.9229353115677913e-012, 5.7323831678020873e-016
            ],
            [
                2.4087011554664056e-001, 1.6145951286700025e-001, 7.2069364017178436e-002, 2.1126344408967029e-002,
                3.9766089291813113e-003, 4.6471871877939763e-004, 3.2095005652745989e-005, 1.2176597454425830e-006,
                2.2674616734804651e-008, 1.7186649279648690e-010, 3.7149741527624159e-013, 9.3901936890419202e-017
            ],
            [
                2.4816935117648548e-001, 2.0485102565034041e-001, 1.1488092430395164e-001, 4.3379970167644971e-002,
                1.0856755991462316e-002, 1.7578504052637961e-003, 1.7776690692652660e-004, 1.0672194905202536e-005,
                3.5301525602454978e-007, 5.7380238688993763e-009, 3.7911500004771871e-011, 7.1021030370039253e-014,
                1.5300389979986825e-017
            ]
        ]
    try:
        node_data = np.array(node_data_list[level - 1], np.float64)
        weight_data = np.array(weight_data_list[level - 1], np.float64)
    except IndexError:
        raise ValueError(f"Quadrature rules are only available up to a level of {len(node_data_list)}.")
    return node_data, weight_data

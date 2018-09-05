"""Basic functionality."""

import re
import time
import inspect
import datetime
import contextlib
import multiprocessing.pool
from typing import (
    Any, Tuple, Iterable, Iterator, Callable, Optional, Sequence, Container, Mapping, Union, List, Hashable, Dict
)

import numpy as np

from .. import options


# define common types
Array = Any
RecArray = Any
Data = Dict[str, Array]
Options = Dict[str, Any]
Bounds = Tuple[Array, Array]

# define a pool managed by parallel and used by generate_items
pool = None


@contextlib.contextmanager
def parallel(processes: int) -> Iterator[None]:
    """Context manager used for parallel processing in a ``with`` statement context.

    This manager creates a a context in which a pool of Python processes will be used by any of the following methods,
    which all support parallel processing:

        - :meth:`Simulation.solve`
        - :meth:`Problem.solve`
        - Any method in :class:`Results`.

    These methods, which perform market-by-market computation, will distribute their work among the processes.
    After the context created by the ``with`` statement ends, all worker processes in the pool will be terminated.
    Outside of this context, such methods will not use multiprocessing.

    Importantly, multiprocessing will only improve speed if gains from parallelization outweigh overhead from
    serializing and passing data between processes. For example, if computation for a single market is very fast and
    there is a lot of data in each market that must be serialized and passed between processes, using multiprocessing
    may reduce overall speed.

    Arguments
    ---------
    processes : `int`
        Number of Python processes that will be created and used by any method that supports parallel processing.

    Example
    -------
    The following code uses multiprocessing to compute elasticities market-by-market for a simple Logit problem
    configured with some of the fake cereal data from :ref:`Nevo (2000) <n00>`.

    .. ipython:: python

       product_data = np.recfromcsv(pyblp.data.NEVO_PRODUCTS_LOCATION, encoding='utf-8')
       formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
       problem = pyblp.Problem(formulation, product_data)
       results = problem.solve()
       with pyblp.parallel(2):
            elasticities = results.compute_elasticities()
       elasticities

    Solving a Logit problem does not require market-by-market computation, so parallelization does not change its
    estimation procedure. Although elasticity computation does happen market-by-market, this problem is very small, so
    there are no gains from parallelization in this example.

    If the problem were much larger, running :meth:`Problem.solve` and :meth:`Results.compute_elasticities` under the
    ``with`` statement could substantially speed up estimation and elasticity computation.

    For more examples, refer to the :doc:`Examples </examples>` section.

    """

    # validate the number of processes
    if not isinstance(processes, int):
        raise TypeError("processes must be an int.")
    if processes < 2:
        raise ValueError("processes must be at least 2.")

    # start the process pool, wait for work to be done, and then terminate it
    output(f"Starting a pool of {processes} processes ...")
    start_time = time.time()
    global pool
    try:
        with multiprocessing.pool.Pool(processes) as pool:
            output(f"Started the process pool after {format_seconds(time.time() - start_time)}.")
            yield
            output(f"Terminating the pool of {processes} processes ...")
            terminate_time = time.time()
    finally:
        pool = None
    output(f"Terminated the process pool after {format_seconds(time.time() - terminate_time)}.")


def generate_items(keys: Iterable, factory: Callable[[Any], tuple], method: Callable) -> Iterator:
    """Generate (key, method(*factory(key))) tuples for each key. The first element returned by factory is an instance
    of the class to which method is attached. If a process pool has been initialized, use multiprocessing; otherwise,
    use serial processing.
    """
    if pool is None:
        return (generate_items_worker((k, factory(k), method)) for k in keys)
    return pool.imap_unordered(generate_items_worker, ((k, factory(k), method) for k in keys))


def generate_items_worker(args: Tuple[Any, tuple, Callable]) -> Tuple[Any, Any]:
    """Call the the specified method of a class instance with any additional arguments. Return the associated key along
    with the returned object.
    """
    key, (instance, *method_args), method = args
    return key, method(instance, *method_args)


def structure_matrices(mapping: Mapping) -> RecArray:
    """Structure a mapping of keys to (array or None, type) tuples as a record array in which each sub-array is
    guaranteed to be at least two-dimensional.
    """

    # determine the number of rows in all matrices
    size = next(a.shape[0] for a, _ in mapping.values() if a is not None)

    # collect matrices and data types
    matrices: List[Array] = []
    dtypes: List[Tuple[Union[str, Tuple[Hashable, str]], Any, Tuple[int]]] = []
    for key, (array, dtype) in mapping.items():
        matrix = np.zeros((size, 0)) if array is None else np.c_[array]
        dtypes.append((key, dtype, (matrix.shape[1],)))
        matrices.append(matrix)

    # build the record array
    structured = np.recarray(size, dtypes)
    for dtype, matrix in zip(dtypes, matrices):
        structured[dtype[0] if isinstance(dtype[0], str) else dtype[0][1]] = matrix
    return structured


def extract_matrix(structured_array_like: Mapping, key: Any) -> Optional[Array]:
    """Attempt to extract a field from a structured array-like object or horizontally stack field0, field1, and so on,
    into a full matrix. The extracted array will have at least two dimensions.
    """
    try:
        matrix = np.c_[structured_array_like[key]]
        return matrix if matrix.size > 0 else None
    except Exception:
        index = 0
        parts: List[Array] = []
        while True:
            try:
                part = np.c_[structured_array_like[f'{key}{index}']]
            except Exception:
                break
            index += 1
            if part.size > 0:
                parts.append(part)
        return np.hstack(parts) if parts else None


def extract_size(structured_array_like: Mapping) -> int:
    """Attempt to extract the number of rows from a structured array-like object."""
    size = 0
    getters = [
        lambda m: m.shape[0],
        lambda m: next(iter(structured_array_like.values())).shape[0],
        lambda m: len(next(iter(structured_array_like.values()))),
        lambda m: len(m)
    ]
    for get in getters:
        try:
            size = get(structured_array_like)
            break
        except Exception:
            pass
    if size > 0:
        return size
    raise TypeError(
        f"Failed to get the number of rows in the structured array-like object of type {type(structured_array_like)}. "
        f"Try using a dictionary, a NumPy structured array, a Pandas DataFrame, or any other standard type."
    )


def output(message: Any) -> None:
    """Print a message if verbosity is turned on."""
    if options.verbose:
        if not callable(options.verbose_output):
            raise TypeError("options.verbose_output should be callable.")
        options.verbose_output(str(message))


def format_seconds(seconds: float) -> str:
    """Prepare a number of seconds to be displayed as a string."""
    return str(datetime.timedelta(seconds=int(round(seconds))))


def format_number(number: Optional[float]) -> str:
    """Prepare a number to be displayed as a string."""
    if number is None or np.isnan(number):
        return "NA"
    if not isinstance(options.digits, int):
        raise TypeError("options.digits must be an int.")
    template = f"{{:+.{options.digits - 1}E}}"
    return template.format(float(number))


def format_se(se: Optional[float]) -> str:
    """Prepare a standard error to be displayed as a string."""
    return f"({format_number(se)})"


def format_options(mapping: Options) -> str:
    """Prepare a mapping of options to be displayed as a string."""
    strings: List[str] = []
    for key, value in mapping.items():
        if callable(value):
            value = f'{value.__module__}.{value.__qualname__}'
        elif isinstance(value, float):
            value = format_number(value)
        strings.append(f'{key}: {value}')
    joined = ', '.join(strings)
    return f'{{{joined}}}'


class TableFormatter(object):
    """Formatter of tables with fixed-width columns."""

    template: str
    widths: List[int]

    def __init__(self, widths: Sequence[int], line_indices: Container[int] = ()) -> None:
        """Build the table's template string, which has fixed widths and vertical lines after specified indices."""
        parts = ["{{:^{}}}{}".format(w, "  |" if i in line_indices else "") for i, w in enumerate(widths)]
        self.template = "  ".join(parts)
        self.widths = list(widths)

    def __call__(self, values: Sequence, underline: bool = False) -> str:
        """Construct a row. If underline is True, construct a second row of underlines."""
        formatted = self.template.format(*map(str, list(values) + [""] * (len(self.widths) - len(values))))
        if underline:
            return "\n".join([formatted, self(["-" * w for w in self.widths[:len(values)]])])
        return formatted

    def blank(self) -> str:
        """Construct a blank row."""
        return self([""] * len(self.widths))

    def line(self) -> str:
        """Construct a horizontal line."""
        return "=" * len(self.blank())


class Groups(object):
    """Computation of grouped statistics."""

    sort: Array
    undo: Array
    unique: Array
    index: Array
    inverse: Array
    counts: Array

    def __init__(self, ids: Array) -> None:
        """Sort and index IDs that define groups."""
        self.sort = ids.flatten().argsort()
        self.undo = self.sort.argsort()
        self.unique, self.index, self.inverse, self.counts = np.unique(
            ids[self.sort], return_index=True, return_inverse=True, return_counts=True
        )

    def sum(self, matrix: Array) -> Array:
        """Compute the sum of each group."""
        return np.add.reduceat(matrix[self.sort], self.index)

    def mean(self, matrix: Array) -> Array:
        """Compute the mean of each group."""
        return self.sum(matrix) / self.counts[:, np.newaxis]

    def expand(self, statistics: Array) -> Array:
        """Expand statistics for each group to the size of the original matrix."""
        return statistics[self.inverse][self.undo]


class Error(Exception):
    """Errors that are indistinguishable from others with the same message, which is parsed from the docstring."""

    def __eq__(self, other: Any) -> bool:
        """Defer to hashes."""
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """Hash this instance such that in collections it is indistinguishable from others with the same message."""
        return hash((type(self).__name__, str(self)))

    def __repr__(self) -> str:
        """Defer to the string representation."""
        return str(self)

    def __str__(self) -> str:
        """Replace docstring markdown with simple text."""
        doc = inspect.getdoc(self)

        # normalize LaTeX
        while True:
            match = re.search(r':math:`([^`]+)`', doc)
            if not match:
                break
            start, end = match.span()
            doc = doc[:start] + re.sub(r'\s+', ' ', re.sub(r'[\\{}]', ' ', match.group(1))).lower() + doc[end:]

        # normalize references
        while True:
            match = re.search(r':ref:`([^`]+)`', doc)
            if not match:
                break
            start, end = match.span()
            doc = doc[:start] + re.sub(r'<[^>]+>', '', match.group(1)) + doc[end:]

        # remove all remaining domains and compress whitespace
        return re.sub(r'[\s\n]+', ' ', re.sub(r':[a-z\-]+:|`', '', doc))

"""Basic functionality."""

import contextlib
import functools
import inspect
import multiprocessing.pool
import re
import sys
import time
import traceback
from typing import (
    Any, Callable, Container, Dict, Hashable, Iterable, Iterator, List, Mapping, Optional, Set, Sequence, Type, Tuple,
    Union
)
import warnings

import numpy as np

from .. import options


# define common types
Array = Any
RecArray = Any
Data = Dict[str, Array]
Options = Dict[str, Any]
Bounds = Tuple[Array, Array]

# define pools managed by parallel and used by generate_items
pool: Any = None


@contextlib.contextmanager
def parallel(processes: int, use_pathos: bool = False) -> Iterator[None]:
    r"""Context manager used for parallel processing in a ``with`` statement context.

    This manager creates a context in which a pool of Python processes will be used by any method that requires
    market-by-market computation. These methods will distribute their work among the processes. After the context
    created by the ``with`` statement ends, all worker processes in the pool will be terminated. Outside this context,
    such methods will not use multiprocessing.

    Importantly, multiprocessing will only improve speed if gains from parallelization outweigh overhead from
    serializing and passing data between processes. For example, if computation for a single market is very fast and
    there is a lot of data in each market that must be serialized and passed between processes, using multiprocessing
    may reduce overall speed.

    Arguments
    ---------
    processes : `int`
        Number of Python processes that will be created and used by any method that supports parallel processing.
    use_pathos : `bool, optional`
        Whether to use `pathos <https://pathos.readthedocs.io/en/latest/>`_ (which will need to be installed) instead of
        the default, built-in :mod:`multiprocessing` module.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/parallel.ipynb

    .. raw:: latex

       \end{examplenotebook}

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
    if use_pathos:
        try:
            from pathos.multiprocessing import ProcessPool
        except ImportError as exception:
            if "pathos" not in str(exception):
                raise
            raise ImportError("pathos must be installed when use_pathos is True.") from exception
        try:
            pool = ProcessPool(nodes=processes)
            output(f"Started the process pool after {format_seconds(time.time() - start_time)}.")
            yield
        finally:
            output(f"Terminating the pool of {processes} processes ...")
            terminate_time = time.time()
            try:
                pool.close()
            except Exception:
                pass
            try:
                pool.join()
            except Exception:
                pass
            try:
                pool.clear()
            except Exception:
                pass
            pool = None
    else:
        try:
            with multiprocessing.pool.Pool(processes) as pool:
                output(f"Started the process pool after {format_seconds(time.time() - start_time)}.")
                yield
                output(f"Terminating the pool of {processes} processes ...")
                terminate_time = time.time()
        except AttributeError as exception:
            if "Can't pickle local object" not in str(exception) or "<lambda>" not in str(exception):
                raise
            pathos_message = (
                "The built-in multiprocessing module does not support lambda functions. Consider setting "
                "the use_pathos of parallel to True."
            )
            raise RuntimeError(pathos_message) from exception
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
    try:
        return pool.imap_unordered(generate_items_worker, ((k, factory(k), method) for k in keys))
    except AttributeError:
        # a pathos ProcessPool uses uimap instead of imap_unordered
        return pool.uimap(generate_items_worker, ((k, factory(k), method) for k in keys))


def generate_items_worker(args: Tuple[Any, tuple, Callable]) -> Tuple[Any, Any]:
    """Call the specified method of a class instance with any additional arguments. Return the associated key along with
    the returned object.
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
        dtypes.append((key, dtype, matrix.shape[1:]))
        matrices.append(matrix)

    # build the record array
    structured: RecArray = np.recarray(size, dtypes)
    for dtype, matrix in zip(dtypes, matrices):
        structured[dtype[0] if isinstance(dtype[0], str) else dtype[0][1]] = matrix
    return structured


def update_matrices(matrices: RecArray, update_mapping: Dict) -> RecArray:
    """Update fields in a record array created by structure_matrices by re-structuring the matrices."""
    mapping = update_mapping.copy()
    for key in matrices.dtype.names:
        if key not in mapping:
            if len(matrices.dtype.fields[key]) > 2:
                mapping[(matrices.dtype.fields[key][2], key)] = (matrices[key], matrices[key].dtype)
            else:
                mapping[key] = (matrices[key], matrices[key].dtype)

    return structure_matrices(mapping)


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
                # warn if there's a 1 but no 0 (this is a common mistake)
                if index == 0:
                    try:
                        structured_array_like[f'{key}1']
                    except Exception:
                        pass
                    else:
                        warn(f"'{key}1' was specified but not '{key}0'.")
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


def interact_ids(*columns: Array) -> Array:
    """Create interactions of ID columns."""
    interacted = columns[0].flatten().astype(np.object_)
    if len(columns) > 1:
        interacted[:] = list(zip(*columns))
    return interacted


def warn(message: Any) -> None:
    """Output a warning."""
    old_formatwarning = warnings.formatwarning
    warnings.formatwarning = lambda x, *_, **__: f"{x}\n"
    warnings.warn(message)
    warnings.formatwarning = old_formatwarning


def output(message: Any) -> None:
    """Print a message if verbosity is turned on."""
    if options.verbose:
        if not callable(options.verbose_output):
            raise TypeError("options.verbose_output should be callable.")
        options.verbose_output(str(message))
        if options.flush_output:
            sys.stdout.flush()


def output_progress(iterable: Iterable, length: int, start_time: float) -> Iterator:
    """Yield results from an iterable while outputting progress updates at most every minute."""
    elapsed = time.time() - start_time
    next_minute = int(elapsed / 60) + 1
    for index, iterated in enumerate(iterable):
        yield iterated
        elapsed = time.time() - start_time
        if elapsed > 60 * next_minute:
            output(f"Finished {index + 1} out of {length} after {format_seconds(elapsed)}.")
            next_minute = int(elapsed / 60) + 1


def format_seconds(seconds: float) -> str:
    """Prepare a number of seconds to be displayed as a string."""
    hours, remainder = divmod(int(round(seconds)), 60**2)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def format_number(number: Any) -> str:
    """Prepare a number to be displayed as a string."""
    if not isinstance(options.digits, int):
        raise TypeError("options.digits must be an int.")
    template = f"{{:^+{options.digits + 6}.{options.digits - 1}E}}"
    formatted = template.format(float(number))
    if "NAN" in formatted:
        formatted = formatted.replace("+", " ")
    return formatted


def format_se(se: Any) -> str:
    """Prepare a standard error to be displayed as a string."""
    formatted = format_number(se)
    for string in ["NAN", "-INF", "+INF"]:
        if string in formatted:
            return formatted.replace(string, f"({string})")

    return f"({formatted})"


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


def format_table(
        header: Sequence, *data: Sequence, title: Optional[str] = None, include_border: bool = True,
        include_header: bool = True, line_indices: Container[int] = ()) -> str:
    """Format table information as a string, which has fixed widths, vertical lines after any specified indices, and
    optionally a title, border, and header.
    """

    # construct the header rows
    row_index = -1
    header_rows: List[List[str]] = []
    header = [[c] if isinstance(c, str) else c for c in header]
    while True:
        header_row = ["" if len(c) < -row_index else c[row_index] for c in header]
        if not any(header_row):
            break
        header_rows.insert(0, header_row)
        row_index -= 1

    # construct the data rows
    data_rows = [[str(c) for c in r] + [""] * (len(header) - len(r)) for r in data]

    # compute column widths
    widths = []
    for column_index in range(len(header)):
        widths.append(max(len(r[column_index]) for r in header_rows + data_rows))

    # build the template
    template = "  " .join("{{:^{}}}{}".format(w, "  |" if i in line_indices else "") for i, w in enumerate(widths))

    # build the table
    lines = []
    if title is not None:
        lines.append(f"{title}:")
    if include_border:
        lines.append("=" * len(template.format(*[""] * len(widths))))
    if include_header:
        lines.extend([template.format(*r) for r in header_rows])
        lines.append(template.format(*("-" * w for w in widths)))
    lines.extend([template.format(*r) for r in data_rows])
    if include_border:
        lines.append("=" * len(template.format(*[""] * len(widths))))
    return "\n".join(lines)


def get_indices(ids: Array) -> Dict[Hashable, Array]:
    """From a one-dimensional array input, construct a dictionary with keys that are the unique values of the array
    and values that are the indices where the key appears in the array.
    """
    flat = ids.flatten()
    sort_indices = flat.argsort(kind='mergesort')
    sorted_ids = flat[sort_indices]
    changes = np.ones(flat.shape, np.bool_)
    changes[1:] = sorted_ids[1:] != sorted_ids[:-1]
    reduce_indices = np.nonzero(changes)[0]
    return dict(zip(sorted_ids[reduce_indices], np.split(sort_indices, reduce_indices)[1:]))


def compute_finite_differences(f: Callable[[Array], Array], x: Array, epsilon_scale: float = 1.0) -> Array:
    """Approximate derivatives with finite differences."""
    epsilon = epsilon_scale * options.finite_differences_epsilon

    arrays = []
    for index in range(x.size):
        x1 = x.copy()
        x2 = x.copy()
        x1[index] += epsilon / 2
        x2[index] -= epsilon / 2
        arrays.append((f(x1) - f(x2)) / epsilon)

    if len(arrays[0].shape) == 1 or (len(arrays[0].shape) == 2 and arrays[0].shape[1] == 1):
        return np.column_stack(arrays)
    return np.dstack(arrays)


def compute_second_finite_differences(f: Callable[[Array], Array], x: Array, epsilon_scale: float = 1.0) -> Array:
    """Approximate second derivatives with finite differences."""
    epsilon = np.sqrt(epsilon_scale * options.finite_differences_epsilon)

    arrays = []
    for index1 in range(x.size):
        arrays1 = []
        for index2 in range(x.size):
            x1 = x.copy()
            x2 = x.copy()
            x3 = x.copy()
            x4 = x.copy()
            x1[index1] += epsilon / 2
            x2[index1] += epsilon / 2
            x3[index1] -= epsilon / 2
            x4[index1] -= epsilon / 2
            x1[index2] += epsilon / 2
            x2[index2] -= epsilon / 2
            x3[index2] += epsilon / 2
            x4[index2] -= epsilon / 2
            arrays1.append((f(x1) - f(x2) - f(x3) + f(x4)) / epsilon**2)

        if len(arrays1[0].shape) == 1 or (len(arrays1[0].shape) == 2 and arrays1[0].shape[1] == 1):
            arrays.append(np.column_stack(arrays1))
        else:
            arrays.append(np.dstack(arrays1))

    return np.dstack(arrays)


class SolverStats(object):
    """Structured statistics returned by a generic numerical solver."""

    converged: bool
    iterations: int
    evaluations: int

    def __init__(self, converged: bool = True, iterations: int = 0, evaluations: int = 0) -> None:
        """Structure the statistics."""
        self.converged = converged
        self.iterations = iterations
        self.evaluations = evaluations


class StringRepresentation(object):
    """Object that defers to its string representation."""

    def __repr__(self) -> str:
        """Defer to the string representation."""
        return str(self)


class Groups(object):
    """Computation of grouped statistics."""

    sort_indices: Array
    reduce_indices: Array
    unique: Array
    codes: Array
    counts: Array
    group_count: int

    def __init__(self, ids: Array) -> None:
        """Sort and index IDs that define groups."""

        # sort the IDs
        flat = ids.flatten()
        self.sort_indices = flat.argsort()
        sorted_ids = flat[self.sort_indices]

        # identify groups
        changes = np.ones(flat.shape, np.bool_)
        changes[1:] = sorted_ids[1:] != sorted_ids[:-1]
        self.reduce_indices = np.nonzero(changes)[0]
        self.unique = sorted_ids[self.reduce_indices]

        # encode the groups
        sorted_codes: Array = np.cumsum(changes) - 1
        self.codes = sorted_codes[self.sort_indices.argsort()]

        # compute counts
        self.group_count = self.reduce_indices.size
        self.counts = np.diff(np.append(self.reduce_indices, self.codes.size))

    def sum(self, matrix: Array) -> Array:
        """Compute the sum of each group."""
        return np.add.reduceat(matrix[self.sort_indices], self.reduce_indices)

    def mean(self, matrix: Array) -> Array:
        """Compute the mean of each group."""
        return self.sum(matrix) / self.counts[:, None]

    def expand(self, statistics: Array) -> Array:
        """Expand statistics for each group to the size of the original matrix."""
        return statistics[self.codes]


class Error(Exception):
    """Errors that are indistinguishable from others with the same message, which is parsed from the docstring."""

    stack: Optional[str]

    def __init__(self) -> None:
        """Optionally store the full current traceback for debugging purposes."""
        if options.verbose_tracebacks:
            self.stack = ''.join(traceback.format_stack())
        else:
            self.stack = None

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
        assert doc is not None

        # normalize LaTeX
        while True:
            match = re.search(r':math:`([^`]+)`', doc)
            if match is None:
                break
            start, end = match.span()
            doc = doc[:start] + re.sub(r'\s+', ' ', re.sub(r'[\\{}]', ' ', match.group(1))).lower() + doc[end:]

        # normalize references
        while True:
            match = re.search(r':ref:`[a-zA-Z0-9]+:([^`]+)`', doc)
            if match is None:
                break
            start, end = match.span()
            doc = doc[:start] + re.sub(r'<[^>]+>', '', match.group(1)) + doc[end:]

        # remove all remaining domains and compress whitespace
        doc = re.sub(r'[\s\n]+', ' ', re.sub(r':[a-z\-]+:|`', '', doc))

        # optionally add the full traceback
        if self.stack is not None:
            doc = f"{doc} Traceback:\n\n{self.stack}\n"
        return doc


class DerivedError(Error):
    """Error derived from another exception."""

    _exception: Exception

    def __init__(self, exception: Exception) -> None:
        """Store the exception from which this error is derived."""
        super().__init__()
        self._exception = exception

    def __str__(self) -> str:
        """Supplement the error with the exception's message."""
        return f"{super().__str__()} Exception encountered: '{self._exception}'."


class NumericalError(Error):
    """Floating point issues."""

    _messages: Set[str]

    def __init__(self) -> None:
        super().__init__()
        self._messages: Set[str] = set()

    def __str__(self) -> str:
        """Supplement the error with the messages."""
        combined = ", ".join(sorted(self._messages))
        return f"{super().__str__()} Errors encountered: {combined}."


class MultipleReversionError(Error):
    """Reversion of problematic elements."""

    _bad: int
    _total: int

    def __init__(self, bad_indices: Array) -> None:
        """Store element counts."""
        super().__init__()
        self._bad = bad_indices.sum()
        self._total = bad_indices.size

    def __str__(self) -> str:
        """Supplement the error with the counts."""
        return f"{super().__str__()} Number of reverted elements: {self._bad} out of {self._total}."


class InversionError(Error):
    """Problems with inverting a matrix."""

    _condition: float

    def __init__(self, matrix: Array) -> None:
        """Compute condition number of the matrix."""
        super().__init__()
        from .algebra import compute_condition_number
        self._condition = compute_condition_number(matrix)

    def __str__(self) -> str:
        """Supplement the error with the condition number."""
        return f"{super().__str__()} Condition number: {format_number(self._condition)}."


class InversionReplacementError(InversionError):
    """Problems with inverting a matrix led to the use of a replacement such as an approximation."""

    _replacement: str

    def __init__(self, matrix: Array, replacement: str) -> None:
        """Store the replacement description."""
        super().__init__(matrix)
        self._replacement = replacement

    def __str__(self) -> str:
        """Supplement the error with the description."""
        return f"{super().__str__()} The inverse was replaced with {self._replacement}."


class NumericalErrorHandler(object):
    """Decorator that appends errors to a function's returned list when numerical errors are encountered."""

    error: Type[NumericalError]

    def __init__(self, error: Type[NumericalError]) -> None:
        """Store the error class."""
        self.error = error

    def __call__(self, decorated: Callable) -> Callable:
        """Decorate the function."""
        @functools.wraps(decorated)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Configure NumPy to detect numerical errors."""
            detector = NumericalErrorDetector(self.error)
            with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
                np.seterrcall(detector)
                returned = decorated(*args, **kwargs)
            if detector.detected is not None:
                returned[-1].append(detector.detected)
            return returned

        return wrapper


class NumericalErrorDetector(object):
    """Error detector to be passed to NumPy's error call function."""

    error: Type[NumericalError]
    detected: Optional[NumericalError]

    def __init__(self, error: Type[NumericalError]) -> None:
        """By default no error is detected."""
        self.error = error
        self.detected = None

    def __call__(self, message: str, _: int) -> None:
        """Initialize the error and store the error message."""
        if self.detected is None:
            self.detected = self.error()
        self.detected._messages.add(message)

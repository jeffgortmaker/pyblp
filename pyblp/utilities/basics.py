"""Basic functionality."""

import time
import datetime
import contextlib
import multiprocessing

import numpy as np

from .. import options


@contextlib.contextmanager
def parallel(processes):
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
    The following code uses multiprocessing to solve a very simple problem with some of the automobile product data from
    :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`. The same process pool is then used to compute elasticities. Since
    this problem uses a small dataset, there are no gains from parallelization.

    .. ipython:: python

       products = np.recfromcsv(pyblp.data.BLP_PRODUCTS_LOCATION)
       products = {n: products[n] for n in products.dtype.names}
       products['demand_instruments'] = pyblp.build_blp_instruments(
           pyblp.Formulation('hpwt + air + mpg + space'),
           products
       )
       problem = pyblp.Problem(
           product_formulations=(
               pyblp.Formulation('prices + hpwt + air + mpg + space'),
               pyblp.Formulation('hpwt + air')
           ),
           product_data=products,
           integration=pyblp.Integration('monte_carlo', 50, seed=0)
       )
       initial_sigma = np.eye(3)
       with pyblp.parallel(2):
           results = problem.solve(initial_sigma, steps=1)
           elasticities = results.compute_elasticities()
       results
       elasticities

    """

    # validate the number of processes
    if not isinstance(processes, int):
        raise TypeError("processes must be an int.")
    if processes < 2:
        raise ValueError("processes must be at least 2.")

    # start the process pool, wait for work to be done, and then terminate it
    output(f"Starting a pool of {processes} processes ...")
    start_time = time.time()
    with multiprocessing.Pool(processes) as parallel._pool:
        output(f"Started the process pool after {format_seconds(time.time() - start_time)}.")
        yield
        output(f"Terminating the pool of {processes} processes ...")
        terminate_time = time.time()
    del parallel._pool
    output(f"Terminated the process pool after {format_seconds(time.time() - terminate_time)}.")


def generate_items(keys, factory, method):
    """Generate (key, method(*factory(key))) tuples for each key. The first element returned by factory is an instance
    of the class to which method is attached. If parallel._pool is initialized, use multiprocessing; otherwise, use
    serial processing.
    """
    try:
        generate = parallel._pool.imap_unordered
    except AttributeError:
        generate = map
    return generate(generate_items_worker, ((k, factory(k), method) for k in keys))


def generate_items_worker(args):
    """Call the the specified method of a class instance with any additional arguments. Return the associated key along
    with the returned object.
    """
    key, (instance, *method_args), method = args
    return key, method(instance, *method_args)


def extract_matrix(structured_array_like, key):
    """Attempt to extract a field from a structured array-like object or horizontally stack field0, field1, and so on,
    into a full matrix. The extracted array will have at least two dimensions.
    """
    try:
        matrix = np.c_[structured_array_like[key]]
        return matrix if matrix.size > 0 else None
    except:
        index = 0
        parts = []
        while True:
            try:
                part = np.c_[structured_array_like[f'{key}{index}']]
            except:
                break
            index += 1
            if part.size > 0:
                parts.append(part)
        return np.hstack(parts) if parts else None


def extract_size(structured_array_like):
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
        except:
            pass
    if size > 0:
        return size
    raise TypeError(
        f"Failed to get the number of rows in the structured array-like object of type {type(structured_array_like)}. "
        f"Try using a dictionary, a NumPy structured array, a Pandas DataFrame, or any other standard type."
    )


def output(message):
    """Print a message if verbosity is turned on."""
    if options.verbose:
        if not callable(options.verbose_output):
            raise TypeError("options.verbose_output should be callable.")
        options.verbose_output(str(message))


def format_seconds(seconds):
    """Prepare a number of seconds to be displayed as a string."""
    return str(datetime.timedelta(seconds=int(round(seconds))))


def format_number(number):
    """Prepare a number to be displayed as a string."""
    if number is None or np.isnan(number):
        return "NA"
    if not isinstance(options.digits, int):
        raise TypeError("options.digits must be an int.")
    template = f"{{:+.{options.digits - 1}E}}"
    return template.format(float(number))


def format_se(se):
    """Prepare a standard error to be displayed as a string."""
    return f"({format_number(se)})"


class TableFormatter(object):
    """Formatter of tables with fixed-width columns."""

    def __init__(self, widths, line_indices=()):
        """Build the table's template string, which has fixed widths and vertical lines after specified indices."""
        parts = ["{{:^{}}}{}".format(w, "  |" if i in line_indices else "") for i, w in enumerate(widths)]
        self.template = "  ".join(parts)
        self.widths = widths

    def __call__(self, values, underline=False):
        """Construct a row. If underline is True, construct a second row of underlines."""
        formatted = self.template.format(*map(str, list(values) + [""] * (len(self.widths) - len(values))))
        if underline:
            return "\n".join([formatted, self(["-" * w for w in self.widths[:len(values)]])])
        return formatted

    def blank(self):
        """Construct a blank row."""
        return self([""] * len(self.widths))

    def line(self):
        """Construct a horizontal line."""
        return "=" * len(self.blank())


class Groups(object):
    """Computation of grouped statistics."""

    def __init__(self, ids):
        """Sort and index IDs that define groups."""
        self.sort = ids.flatten().argsort()
        self.undo = self.sort.argsort()
        self.unique, self.index, self.inverse, self.counts = np.unique(
            ids[self.sort], return_index=True, return_inverse=True, return_counts=True
        )

    def sum(self, matrix):
        """Compute the sum of each group."""
        return np.add.reduceat(matrix[self.sort], self.index)

    def mean(self, matrix):
        """Compute the mean of each group."""
        return self.sum(matrix) / self.counts[:, np.newaxis]

    def expand(self, statistics):
        """Expand statistics for each group to the size of the original matrix."""
        return statistics[self.inverse][self.undo]


class Matrices(np.recarray):
    """Record array, which guarantees that each sub-array is at least two-dimensional."""

    def __new__(cls, mapping):
        """Construct the record array from a mapping of field keys to (array or None, type) tuples."""

        # determine the number of rows in all matrices
        size = next(a.shape[0] for a, _ in mapping.values() if a is not None)

        # collect data types and matrices
        dtypes = []
        matrices = []
        for key, (array, dtype) in mapping.items():
            matrix = np.zeros((size, 0)) if array is None else np.c_[array]
            dtypes.append((key, dtype, (matrix.shape[1],)))
            matrices.append(matrix)

        # build the record array
        self = np.ndarray.__new__(cls, size, (np.record, dtypes))
        for dtype, matrix in zip(dtypes, matrices):
            self[dtype[0] if isinstance(dtype[0], str) else dtype[0][1]] = matrix
        return self

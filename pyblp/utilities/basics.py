"""Basic functionality."""

import datetime
import itertools
import multiprocessing

import numpy as np


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


class ParallelItems(object):
    """Generator that passes keys to a factory function, which returns a corresponding (instance, *arguments) tuple;
    yields items from a mapping of the keys to objects returned in parallel from a method of the instance, which is
    passed the arguments returned by the factory. Used in a with clause.
    """

    def __init__(self, keys, factory, method, processes):
        """Initialize class attributes."""
        self.keys = keys
        self.factory = factory
        self.method = method
        self.processes = processes
        self.process_objects = []
        self.in_queue = self.out_queue = self.remaining = None

    def __enter__(self):
        """If there is only one process, return a generator that iteratively creates and passes arguments to the method.
        Otherwise, fill an "in" queue with factory-created items, start processes that fill an "out" queue, and return a
        generator that gets method-processed items from the "out" queue.
        """
        if self.processes == 1:
            return ((k, self.method(*self.factory(k))) for k in self.keys)

        # start with an empty "out" queue and fill an "in" queue with keys corresponding to factory-created arguments
        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        for key in self.keys:
            self.in_queue.put((key, self.factory(key)))

        # destroy the factory, which does not need to be pickled
        self.factory = None

        # share the number of remaining items between processes
        self.remaining = multiprocessing.Value('i', len(self.keys))

        # define a function to generate processes that fill the "out" queue by calling this same class (__call__ is used
        #   because the multiprocessing module can only pickle methods in the module-level namespace)
        def generate():
            while True:
                process = multiprocessing.Process(target=self)
                process.daemon = True
                process.start()
                yield process

        # start the processes and return a generator that gets items from the "out" queue
        self.process_objects = list(itertools.islice(generate(), self.processes))
        return (self.out_queue.get() for _ in self.keys)

    def __exit__(self, *_):
        """Terminate any remaining processes."""
        for process in self.process_objects:
            try:
                process.terminate()
            except:
                pass

    def __call__(self):
        """Get items from the "in" queue and put items processed by the method into the "out" queue."""
        while self.remaining.value > 0:
            key, args = self.in_queue.get()
            self.out_queue.put((key, self.method(*args)))
            with self.remaining.get_lock():
                self.remaining.value -= 1


class Output(object):
    """Output standardization."""

    def __init__(self, options):
        """Store options that configure verbosity and the number of digits to display."""
        self.options = options

    def __call__(self, message):
        """Print a message if verbose."""
        if self.options.verbose:
            self.options.verbose_output(str(message))

    @staticmethod
    def table_formatter(widths, line_indices=()):
        """Initialize a TableFormatter."""
        return TableFormatter(widths, line_indices)

    @staticmethod
    def format_seconds(seconds):
        """Prepare a number of seconds to be displayed as a string."""
        return str(datetime.timedelta(seconds=int(round(seconds))))

    def format_number(self, number):
        """Prepare a number to be displayed as a string."""
        if number is None or np.isnan(number):
            return "NA"
        template = f"{{:+.{self.options.digits - 1}E}}"
        return template.format(float(number))

    def format_se(self, se):
        """Prepare a standard error to be displayed as a string."""
        return f"({self.format_number(se)})"


class TableFormatter(object):
    """Formatter of tables with fixed-width columns."""

    def __init__(self, widths, line_indices):
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

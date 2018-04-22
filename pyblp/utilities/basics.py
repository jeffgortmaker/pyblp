"""Basic functionality."""

import datetime
import itertools
import multiprocessing

import numpy as np


def extract_matrix(array, field):
    """Extract a field from an array or horizontally stacks field0, field1, an so on, into a full matrix. The extracted
    array will have at least two dimensions. None is returned if there is no matrix to extract. The array can be a numpy
    structured array, a pandas DataFrame, or anything else that maps strings to array-like objects.
    """
    try:
        return np.c_[array[field]]
    except (KeyError, ValueError):
        index = 0
        parts = []
        while True:
            try:
                parts.append(np.c_[array[f'{field}{index}']])
            except (KeyError, ValueError):
                break
            index += 1
        return np.hstack(parts) if parts else None


class Matrices(np.recarray):
    """Record array, which guarantees that each sub-array is at least two-dimensional."""

    def __new__(cls, mapping):
        """Construct the array from a mapping of field names to (array, type) tuples. None is ignored."""
        names, arrays, types = zip(*((n, a, t) for n, (a, t) in mapping.items() if a is not None))
        dtype = [(n, t, (np.c_[a].shape[1],)) for n, a, t in zip(names, arrays, types)]
        self = np.ndarray.__new__(cls, arrays[0].shape[0], (np.record, dtype))
        for n, a in zip(names, arrays):
            self[n] = np.c_[a]
        return self


class ParallelItems(object):
    """Generator that passes to a worker function in parallel the values of a mapping from keys to lists of arguments.
    Yields items from a mapping of keys to the objects returned by the worker function. Used in a with clause.
    """

    def __init__(self, worker, mapping, processes):
        """Initialize class attributes."""
        self.worker = worker
        self.mapping = mapping
        self.processes = processes
        self.process_objects = []
        self.in_queue = self.out_queue = self.remaining = None

    def __enter__(self):
        """If there is only one process, return a generator that directly calls the worker function. Otherwise, fill an
        "in" queue with items from the mapping, start processes that fill an "out" queue, and return a generator that
        gets items from the "out" queue.
        """
        if self.processes == 1:
            return ((k, self.worker(*a)) for k, a in self.mapping.items())

        # start with an empty "out" queue and an "in" queue filled with the mapping's items
        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        for key, args in self.mapping.items():
            self.in_queue.put((key, args))

        # share the number of remaining items between processes
        total = len(self.mapping)
        self.remaining = multiprocessing.Value('i', total)

        # define a function to generate processes that fill the "out" queue by calling this same class (__call__ is used
        #   because the multiprocessing module can only pickle methods in the module-level namespace)
        def generate():
            while True:
                process = multiprocessing.Process(target=self)
                process.daemon = True
                process.start()
                yield process

        # start the processes and return a generator that gets items from the "out" queue (delete all items from the
        #   mapping so that only the data in the "in" queue are passed to all processes)
        self.mapping = {}
        self.process_objects = list(itertools.islice(generate(), self.processes))
        return (self.out_queue.get() for _ in range(total))

    def __exit__(self, *_):
        """Terminate any remaining processes."""
        for process in self.process_objects:
            try:
                process.terminate()
            except:
                pass

    def __call__(self):
        """Get items from the "in" queue and put items returned by the worker function into the "out" queue."""
        while self.remaining.value > 0:
            key, args = self.in_queue.get()
            self.out_queue.put((key, self.worker(*args)))
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
            print(str(message))

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
    """Format tables with fixed-width columns."""

    def __init__(self, widths, line_indices):
        """Build the table's template string, which has fixed widths and vertical lines after specified indices."""
        parts = ['{{:^{}}}{}'.format(w, '  |' if i in line_indices else '') for i, w in enumerate(widths)]
        self.template = '  '.join(parts)
        self.widths = widths

    def __call__(self, values):
        """Construct a row."""
        return self.template.format(*values)

    def blank(self):
        """Construct a blank row."""
        return self([""] * len(self.widths))

    def lines(self):
        """Construct a line with gaps between columns."""
        return self(["-" * w for w in self.widths])

    def border(self):
        """Construct a line without gaps."""
        return "=" * len(self.blank())

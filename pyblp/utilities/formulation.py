"""Formulation of data matrices."""

import token
import operator
import functools

import patsy
import sympy
import numpy as np
import patsy.builtins
from sympy.parsing import sympy_parser

from .basics import extract_size


class Formulation(object):
    """Configuration for the design of a matrix.

    Internally, the :mod:`patsy` package is used to convert data and R-style formulas into matrices. All of the standard
    operators can be used to design complex matrices of factor interactions. However, since factors need to be
    differentiated (for example, when computing elasticities), only the most essential :mod:`patsy` functionality is
    supported.

    No :mod:`patsy` transformations are supported; instead, data associated with variables should generally already be
    transformed. However, the basic mathematical operations are supported: addition, subtraction, multiplication,
    division, exponentiation, and logarithms.

    Internally, mathematical operations are parsed and evaluated by the :mod:`sympy` package, which is also used to
    symbolically differentiate terms when their derivatives are required (for example, when computing elasticities).

    Parameters
    ----------
    formula : `str`
        R-style formula used to design a matrix. Variable names will be validated when this formulation along with
        data are passed to a function that uses them. By default, an intercept is included, which can be removed with
        ``0`` or ``-1``.

        A small number of functions are included:

            - ``C`` - Mark a variable as categorical. See :func:`patsy.builtins.C`. Arguments are not supported.
            - ``I`` - Encapsulate mathematical operations. See :func:`patsy.builtins.I`.
            - ``log`` - Natural logarithm function.
            - ``exp`` - Natural exponential function.

        All `binary operators <https://patsy.readthedocs.io/en/stable/formulas.html#operators>` are supported:

            - ``+`` - Set union of terms.
            - ``-`` - Set difference of terms.
            - ``*`` - Short-hand. The formula ``a * b`` is the same as ``a + b + a:b``.
            - ``/`` - Short-hand. The formula ``a / b`` is the same as ``a + a:b``.
            - ``:`` - Interactions between two sets of terms.
            - ``**`` - Interactions up to an integer degree.

        When encapsulated by ``I()``, these operators function like normal mathematical operators on numeric variables:
        ``+`` adds, ``-`` subtracts, ``*`` multiplies, ``/`` divides, and ``**`` exponentiates.

    Examples
    --------
    The following code designs a matrix without an intercept, but with both prices and another numeric size variable:

    .. ipython:: python

       formulation = pyblp.Formulation('0 + prices + size')
       formulation

    The following code designs a second matrix with an intercept, with first- and second-degree size terms, with
    categorical product IDs and years, and with the interaction of the last two:

    .. ipython:: python

       formulation = pyblp.Formulation('size + I(size ** 2) + C(product) * C(year)')
       formulation

    The following code designs a third matrix with an intercept and with a yearly trend interacted with the natural
    logarithm of income and categorical education:

    .. ipython:: python

       formulation = pyblp.Formulation('year:(log(income) + C(education))')
       formulation

    """

    def __init__(self, formula):
        """Parse the formula into patsy terms and SymPy expressions. In the process, validate it as much as possible
        without any data.
        """
        if not isinstance(formula, str):
            raise TypeError("formula must be a string.")
        self._terms = parse_terms(formula)
        self._expressions = [parse_term_expression(t) for t in self._terms]
        self._names = {str(s) for e in self._expressions for s in e.free_symbols}
        if not self._terms:
            raise ValueError(f"The formula '{formula}' has no terms.")
        if len(self._terms) > 1 and not any(e.free_symbols for e in self._expressions):
            raise ValueError(f"The formula '{formula}' has more than one term but no variables.")

    def __str__(self):
        """Format the terms as a string."""
        return ' + '.join('1' if t == patsy.desc.INTERCEPT else t.name() for t in self._terms)

    def __repr__(self):
        """Defer to the string representation."""
        return str(self)

    def _build(self, data):
        """Convert a mapping from variable names to arrays into the designed matrix, a list of column formulations that
        describe the columns of the matrix, and a mapping from variable names to arrays of data underlying the matrix,
        which include unchanged continuous variables and indicators constructed from categorical variables.
        """

        # convert the data into a dictionary for convenience (always have at least one column to represent its size)
        data = {n: np.asarray(data[n]) for n in self._names} if self._names else {None: np.zeros(extract_size(data))}

        # design the matrix
        design = design_matrix(self._terms, data)

        # store matrix column indices and build column formulations for each designed column
        column_indices = []
        column_formulations = []
        for term, expression in zip(self._terms, self._expressions):
            term_slice = design.term_slices[term]
            for index in range(term_slice.start, term_slice.stop):
                column_indices.append(index)
                formula = '1' if term == patsy.desc.INTERCEPT else design.column_names[index]
                column_formulations.append(ColumnFormulation(formula, expression))

        # construct a mapping from continuous variable names that appear in at least one column to their arrays
        underlying_data = {}
        for formulation in column_formulations:
            for symbol in formulation.expression.free_symbols:
                underlying_data[symbol.name] = data.get(symbol.name)

        # supplement the mapping with indicators constructed from categorical variables
        for factor, info in design.factor_infos.items():
            if info.type == 'categorical':
                indicator_design = design_matrix([patsy.desc.Term([factor])], data)
                indicator_matrix = build_matrix(indicator_design, data)
                for name, indicator in zip(indicator_design.column_names, indicator_matrix.T):
                    symbol = CategoricalTreatment.parse_full_symbol(name)
                    if symbol.name in underlying_data:
                        underlying_data[symbol.name] = indicator

        # build the matrix
        matrix = build_matrix(design, data)
        return matrix[:, column_indices], column_formulations, underlying_data


class ColumnFormulation(object):
    """Information about a single column in a matrix formulation."""

    def __init__(self, formula, expression):
        """Parse the column name into a patsy term and replace any categorical variables in its SymPy expression with
        the correct indicator variable symbols.
        """
        self.expression = expression
        self.names = {str(s) for s in expression.free_symbols}

        # replace categorical variable symbols with their indicator counterparts
        for factor in parse_terms(formula)[-1].factors:
            name = factor.name()
            base_symbol = CategoricalTreatment.parse_base_symbol(name)
            if base_symbol is not None:
                self.expression = self.expression.replace(base_symbol, CategoricalTreatment.parse_full_symbol(name))

    def __str__(self):
        """Format the expression as a string."""
        return str(self.expression)

    def __repr__(self):
        """Defer to the string representation."""
        return str(self)

    def evaluate(self, data):
        """Evaluate the SymPy column expression at the values supplied by the mapping from variable names to arrays."""
        return evaluate_expression(self.expression, data)

    def evaluate_derivative(self, name, data, order=1):
        """Differentiate the SymPy column expression with respect to a variable name and evaluate the derivative at
        values supplied by the mapping from variable names to arrays.
        """
        return evaluate_expression(self.differentiate(name, order), data)

    def differentiate(self, name, order=1):
        """Differentiate the SymPy column expression with respect to a variable name."""
        return self.expression.diff(sympy.Symbol(name), order)


class EvaluationEnvironment(patsy.eval.EvalEnvironment):
    """Execution environment that parses SymPy expressions from strings and evaluates them at values from a data mapping
    represented as a namespace.
    """

    def subset(self, _):
        """Override the default patsy behavior to create a new patsy evaluation environment instead of a new instance
        of this same class.
        """
        return self.__class__(self._namespaces, self.flags)

    def eval(self, string, **_):
        """Parse a SymPy expression from a string and evaluate it at data represented as the environment's only
        namespace.
        """
        data = self._namespaces[0].copy()

        # parse the SymPy expression, preserving the function that marks variables as categorical
        expression = parse_expression(string, mark_categorical=True)

        # replace categorical variables with unicode objects and explicitly mark them as categorical so that labels are
        #   unique and so that all categorical variables are treated the same
        C = sympy.Function('C')
        for symbol in expression.free_symbols:
            if not np.issubdtype(data[symbol.name].dtype, getattr(np, 'number')):
                expression = expression.replace(symbol, C(symbol))
                data[symbol.name] = data[symbol.name].astype(np.unicode_).astype(np.object_)

        # evaluate the expression and handle universally-marked categorical variables with a non-default coding class
        function_mapping = {'C': functools.partial(patsy.builtins.C, contrast=CategoricalTreatment)}
        evaluated = evaluate_expression(expression, self._namespaces[0], function_mapping)

        # if the evaluated expression is a float or integer, it is a constant that needs to be repeated
        if isinstance(evaluated, (float, int)):
            size = next(iter(data.values())).shape[0]
            evaluated = np.ones(size) * evaluated
        return evaluated


class CategoricalTreatment(patsy.contrasts.Treatment):
    """Standard treatment coding with universal level naming and additional functionality used to parse SymPy symbols
    from categorical variable names that include levels.
    """

    @classmethod
    def code_with_intercept(cls, levels):
        """Code a full-rank contrast matrix."""
        return patsy.contrasts.ContrastMatrix(np.eye(len(levels)), [cls.format_suffix(l) for l in levels])

    @classmethod
    def code_without_intercept(cls, levels):
        """Code a reduced-rank contrast matrix. Choose the first level as the reference level."""
        contrast = cls.code_with_intercept(levels[1:])
        contrast.matrix = np.r_[np.zeros((1, contrast.matrix.shape[1])), contrast.matrix]
        return contrast

    @staticmethod
    def format_suffix(level):
        """Format a level as an indicator variable suffix. Using the level's string representation guarantees a
        one-to-one mapping between indicator variables and their names because categorical variables have already been
        normalized as string objects. It also guarantees that the indicator variable name can be represented as SymPy
        symbol.
        """
        return f'[{level!r}]'

    @staticmethod
    def remove_suffix(name):
        """Drop the suffix from an indicator variable name."""
        return name[:name.index('[')] if '[' in name else None

    @staticmethod
    def extract_suffix(name):
        """Keep only the suffix from an indicator variable name."""
        return name[name.index('['):] if '[' in name else None

    @classmethod
    def parse_base_symbol(cls, name):
        """Parse a SymPy symbol from an indicator variable name that represents the original categorical variable. The
        base name needs to be parsed because it could be encapsulated by the function that explicitly marks categorical
        variables.
        """
        base = cls.remove_suffix(name)
        return None if base is None else parse_expression(base)

    @classmethod
    def parse_full_symbol(cls, name):
        """Parse a SymPy symbol from an indicator variable name that includes the suffix and hence represents the actual
        indicator variable.
        """
        suffix = cls.extract_suffix(name)
        return None if suffix is None else sympy.Symbol(f'{cls.parse_base_symbol(name)}{suffix}')


def parse_terms(formula):
    """Parse patsy terms from a string. Validate that the string contains only right-hand side terms."""
    description = patsy.highlevel.ModelDesc.from_formula(formula)
    if description.lhs_termlist:
        raise ValueError(f"The formula '{formula}' should not have a left-hand side.")
    return description.rhs_termlist


def design_matrix(terms, data):
    """Design a patsy matrix."""
    return patsy.build.design_matrix_builders([terms], lambda: iter([data]), EvaluationEnvironment([data]))[0]


def build_matrix(design, data):
    """Build a matrix according to its design and data mapping variable names to arrays."""

    # identify the number of rows in the data
    size = next(iter(data.values())).shape[0]

    # if the design lacks factors, it must consist of only an intercept term
    if not design.factor_infos:
        return np.ones((size, 1))

    # build the matrix and raise an exception if there are any null values
    matrix = patsy.build.build_design_matrices([design], data, NA_action='raise')[0].base

    # if the design did not use any data, the matrix may be a single row that needs to be stacked to the proper height
    return matrix if matrix.shape[0] == size else np.repeat(matrix[[0]], size, axis=0)


def parse_term_expression(term):
    """Multiply the SymPy expressions parsed from each factor in a patsy term."""
    return functools.reduce(operator.mul, (parse_expression(f.name()) for f in term.factors), sympy.Integer(1))


def parse_expression(string, mark_categorical=False):
    """Parse a SymPy expression from a string. Optionally, preserve the categorical marker function instead of treating
    it like the identify function.
    """

    # list reserved patsy and SymPy names that represent special functions and classes
    patsy_function_names = {'I', 'C'}
    sympy_function_names = {'log', 'exp'}
    sympy_class_names = {'Add', 'Mul', 'Pow', 'Integer', 'Float', 'Symbol'}

    # build a mapping from reserved names to the functions and classes that they represent (patsy functions are dealt
    #   with after parsing)
    mapping = {n: sympy.Function(n) for n in patsy_function_names}
    mapping.update({n: getattr(sympy, n) for n in sympy_function_names | sympy_class_names})

    # define a function that validates a list of tokens into which the string is broken, and that also adds any
    #   unrecognized names as new SymPy symbols
    def transform_tokens(tokens, *_):
        transformed = []
        symbol_candidate = None
        for code, value in tokens:
            if code not in {token.NAME, token.OP, token.NUMBER, token.ENDMARKER}:
                raise ValueError(f"The token '{value}' is invalid.")
            if code == token.OP and value not in {'+', '-', '*', '/', '**', '(', ')'}:
                raise ValueError(f"The operation '{value}' is invalid.")
            if code == token.OP and value == '(' and symbol_candidate is not None:
                raise ValueError(f"The function '{symbol_candidate}' is invalid.")
            if code != token.NAME or value in set(mapping) - sympy_class_names:
                transformed.append((code, value))
                symbol_candidate = None
                continue
            if value in sympy_class_names | {'Intercept'}:
                raise ValueError(f"The name '{value}' is invalid.")
            transformed.extend([(token.NAME, 'Symbol'), (token.OP, '('), (token.NAME, repr(value)), (token.OP, ')')])
            symbol_candidate = value
        return transformed

    # define a function that recursively validates that all categorical marker functions in an expression accept only a
    #   single variable argument, and that they are not arguments to other functions
    def validate_categorical(candidate, depth=0, categorical=False):
        if categorical and depth > 1:
            raise ValueError("The C function must not be an argument to another function.")
        for arg in candidate.args:
            if categorical and not isinstance(arg, sympy.Symbol):
                raise ValueError("The C function accepts only a single variable.")
            validate_categorical(arg, depth + 1, candidate.func == mapping['C'])

    # parse the expression, validate it by attempting to represent it as a string, and validate categorical markers
    try:
        expression = sympy_parser.parse_expr(string, mapping, [transform_tokens], evaluate=False)
        str(expression)
        validate_categorical(expression)
    except (TypeError, ValueError) as exception:
        raise ValueError(f"The expression '{string}' is malformed.") from exception

    # replace patsy functions with the identity function, unless categorical variables are to be explicitly marked
    for name in patsy_function_names:
        if name != 'C' or not mark_categorical:
            expression = expression.replace(mapping[name], sympy.Id)
    return expression


def evaluate_expression(expression, data, function_mapping=None):
    """Evaluate a SymPy expression at data mapping variable names to arrays. Optionally, supplement the default suite of
    NumPy functions with a mapping from non-default function names to functions.
    """
    symbols = list(expression.free_symbols)
    columns = (data[s.name] for s in symbols)
    modules = [function_mapping or {}, 'numpy']
    return sympy.lambdify(symbols, expression, modules)(*columns)

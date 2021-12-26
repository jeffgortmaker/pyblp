"""Formulation of data matrices and absorption of fixed effects."""

import functools
import numbers
import token
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
import patsy
import patsy.builtins
import patsy.contrasts
import patsy.desc
import patsy.design_info
import patsy.origin
import sympy as sp
import sympy.parsing.sympy_parser

from .. import exceptions, options
from ..utilities.basics import Array, Data, Error, StringRepresentation, extract_size, interact_ids


class Formulation(StringRepresentation):
    r"""Configuration for designing matrices and absorbing fixed effects.

    Internally, the `patsy <https://patsy.readthedocs.io/en/stable/>`_ package is used to convert data and R-style
    formulas into matrices. All of the standard
    `binary operators <https://patsy.readthedocs.io/en/stable/formulas.html#operators>`_ can be used to design complex
    matrices of factor interactions:

        - ``+`` - Set union of terms.
        - ``-`` - Set difference of terms.
        - ``*`` - Short-hand. The formula ``a * b`` is the same as ``a + b + a:b``.
        - ``/`` - Short-hand. The formula ``a / b`` is the same as ``a + a:b``.
        - ``:`` - Interactions between two sets of terms.
        - ``**`` - Interactions up to an integer degree.

    However, since factors need to be differentiated (for example, when computing elasticities), only the most essential
    functions are supported:

        - ``C`` - Mark a variable as categorical. See :func:`patsy.builtins.C`. Arguments are not supported.
        - ``I`` - Encapsulate mathematical operations. See :func:`patsy.builtins.I`.
        - ``log`` - Natural logarithm function.
        - ``exp`` - Natural exponential function.

    Data associated with variables should generally already be transformed. However, when encapsulated by ``I()``, these
    operators function like normal mathematical operators on numeric variables: ``+`` adds, ``-`` subtracts, ``*``
    multiplies, ``/`` divides, and ``**`` exponentiates.

    Internally, mathematical operations are parsed and evaluated by the `SymPy <https://www.sympy.org/en/index.html>`_
    package, which is also used to symbolically differentiate terms when derivatives are needed.

    Parameters
    ----------
    formula : `str`
        R-style formula used to design a matrix. Variable names will be validated when this formulation and data are
        passed to a function that uses them. By default, an intercept is included, which can be removed with ``0`` or
        ``-1``. If ``absorb`` is specified, intercepts are ignored.
    absorb : `str, optional`
        R-style formula used to design a matrix of categorical variables representing fixed effects, which will be
        absorbed into the matrix designed by ``formula`` by the `PyHDFE <https://pyhdfe.readthedocs.io/en/stable/>`_
        package. Fixed effect absorption is only supported for some matrices. Unlike ``formula``, intercepts are
        ignored. Only categorical variables are supported.
    absorb_method : `str, optional`
        Method by which fixed effects will be absorbed. For a full list of supported methods, refer to the
        ``residualize_method`` argument of :func:`pyhdfe.create`.

        By default, the simplest methods are used: simple de-meaning for a single fixed effect and simple iterative
        de-meaning by way of the method of alternating projections (MAP) for multiple dimensions of fixed effects. For
        multiple dimensions, non-accelerated MAP is unlikely to be the fastest algorithm. If fixed effect absorption
        seems to be taking a long time, consider using a different method such as ``'lsmr'``, using ``absorb_options``
        to specify a MAP acceleration method, or configuring other options such as termination tolerances.

    absorb_options : `dict, optional`
        Configuration options for the chosen ``method``, which will be passed to the ``options`` argument of
        :func:`pyhdfe.create`.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/formulation.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    _formula: str
    _absorb: Optional[str]
    _absorb_method: Optional[str]
    _absorb_options: dict
    _terms: List[patsy.desc.Term]
    _absorbed_terms: List[patsy.desc.Term]
    _expressions: List[sp.Expr]
    _absorbed_expressions: List[sp.Expr]
    _names: Set[str]
    _absorbed_names: Set[str]

    def __init__(
            self, formula: str, absorb: Optional[str] = None, absorb_method: Optional[str] = None,
            absorb_options: Optional[Mapping] = None) -> None:
        """Parse the formula into patsy terms and SymPy expressions. In the process, validate it as much as possible
        without any data.
        """

        # validate the formulas
        if not isinstance(formula, str):
            raise TypeError("formula must be a str.")
        if absorb is not None and not isinstance(absorb, str):
            raise TypeError("absorb must be a None or a str.")

        # parse the formulas into patsy terms
        self._formula = formula
        self._absorb = absorb
        self._terms = parse_terms(formula)
        self._absorbed_terms: List[patsy.desc.Term] = []
        if absorb is not None:
            self._absorbed_terms = parse_terms(f'{absorb} - 1')

        # ignore intercepts if there are any absorbed terms and check that there is at least one term
        if self._absorbed_terms:
            self._terms = [t for t in self._terms if t != patsy.desc.INTERCEPT]
        if not self._terms:
            raise patsy.PatsyError("formula has no terms.", patsy.origin.Origin(formula, 0, len(formula)))

        # parse the terms into SymPy expressions and extract variable names
        self._expressions = [parse_term_expression(t) for t in self._terms]
        self._absorbed_expressions = [parse_term_expression(t) for t in self._absorbed_terms]
        self._names = {str(s) for e in self._expressions for s in e.free_symbols}
        self._absorbed_names = {str(s) for e in self._absorbed_expressions for s in e.free_symbols}
        if sum(not e.free_symbols for e in self._expressions) > 1:
            origin = patsy.origin.Origin(formula, 0, len(formula))
            raise patsy.PatsyError("formula should have at most one constant term.", origin)
        if self._absorbed_expressions and any(not e.free_symbols for e in self._absorbed_expressions):
            assert absorb is not None
            origin = patsy.origin.Origin(absorb, 0, len(absorb))
            raise patsy.PatsyError("absorb should not have any constant terms.", origin)

        # validate fixed effect absorption options
        if absorb_method is not None and not isinstance(absorb_method, str):
            raise TypeError("absorb_method must be None or a string.")
        if absorb_options is None:
            absorb_options = {}
        elif not isinstance(absorb_options, dict):
            raise TypeError("absorb_options must be None or a dict.")
        self._absorb_method = absorb_method
        self._absorb_options = absorb_options

    def __reduce__(self) -> Tuple[Type['Formulation'], Tuple]:
        """Handle pickling."""
        return (self.__class__, (self._formula, self._absorb, self._absorb_method, self._absorb_options))

    def __str__(self) -> str:
        """Format the terms as a string."""
        names: List[str] = []
        for term in self._terms:
            names.append('1' if term == patsy.desc.INTERCEPT else term.name())
        for absorbed_term in self._absorbed_terms:
            names.append(f'Absorb[{absorbed_term.name()}]')
        return ' + '.join(names)

    def _build_matrix(
            self, data: Mapping, fallback_index: Optional[int] = None) -> (
            Tuple[Array, List['ColumnFormulation'], Data]):
        """Convert a mapping from variable names to arrays into the designed matrix, a list of column formulations that
        describe the columns of the matrix, and a mapping from variable names to arrays of data underlying the matrix,
        which include unchanged continuous variables and indicators constructed from categorical variables. If there is
        a fallback index, allow variables to optionally have this index.
        """

        # normalize the data
        data_mapping: Data = {}
        for name in self._names:
            try:
                data_mapping[name] = np.asarray(data[name]).flatten()
            except Exception as exception:
                fallback = False
                if fallback_index is not None:
                    try:
                        data_mapping[name] = np.asarray(data[f'{name}{fallback_index}']).flatten()
                    except Exception:
                        pass
                    else:
                        fallback = True

                if not fallback:
                    origin = patsy.origin.Origin(self._formula, 0, len(self._formula))
                    message = f"Failed to load data for '{name}' because of the above exception."
                    raise patsy.PatsyError(message, origin) from exception

        # always have at least one column to represent the size of the data
        if not data_mapping:
            data_mapping = {'': np.zeros(extract_size(data))}

        # design the matrix (adding an intercept term if there are absorbed terms gets Patsy to use reduced coding)
        if self._absorbed_terms:
            matrix_design = design_matrix([patsy.desc.INTERCEPT] + self._terms, data_mapping)
        else:
            matrix_design = design_matrix(self._terms, data_mapping)

        # store matrix column indices and build column formulations for each designed column (ignore the intercept if
        #   it was added only to get Patsy to use reduced coding)
        column_indices: List[int] = []
        column_formulations: List[ColumnFormulation] = []
        for term, expression in zip(self._terms, self._expressions):
            if term != patsy.desc.INTERCEPT or not self._absorbed_terms:
                term_slice = matrix_design.term_slices[term]
                for index in range(term_slice.start, term_slice.stop):
                    column_indices.append(index)
                    formula = '1' if term == patsy.desc.INTERCEPT else matrix_design.column_names[index]
                    column_formulations.append(ColumnFormulation(formula, expression))

        # construct a mapping from continuous variable names that appear in at least one column to their arrays
        underlying_data: Data = {}
        for formulation in column_formulations:
            for symbol in formulation.expression.free_symbols:
                underlying_data[symbol.name] = data_mapping.get(symbol.name)

        # supplement the mapping with indicators constructed from categorical variables
        for factor, info in matrix_design.factor_infos.items():
            if info.type == 'categorical':
                indicator_design = design_matrix([patsy.desc.Term([factor])], data_mapping)
                indicator_matrix = build_matrix(indicator_design, data_mapping)
                for name, indicator in zip(indicator_design.column_names, indicator_matrix.T):
                    symbol = CategoricalTreatment.parse_full_symbol(name)
                    if symbol.name in underlying_data:
                        underlying_data[symbol.name] = indicator

        matrix = build_matrix(matrix_design, data_mapping)
        return matrix[:, column_indices], column_formulations, underlying_data

    def _build_ids(self, data: Mapping) -> Array:
        """Convert a mapping from variable names to arrays into the designed matrix of IDs to be absorbed."""

        # normalize the data
        data_mapping: Data = {}
        for name in self._absorbed_names:
            try:
                data_mapping[name] = np.asarray(data[name]).flatten()
            except Exception as exception:
                assert self._absorb is not None
                origin = patsy.origin.Origin(self._absorb, 0, len(self._absorb))
                message = f"Failed to load data for '{name}' because of the above exception."
                raise patsy.PatsyError(message, origin) from exception

        # build columns of absorbed IDs
        ids_columns: List[Array] = []
        for term in self._absorbed_terms:
            factor_columns: List[Array] = []
            term_design = design_matrix([term], data_mapping)
            for factor, info in term_design.factor_infos.items():
                if info.type != 'categorical':
                    raise patsy.PatsyError("Only categorical variables can be absorbed.", factor.origin)
                symbol = parse_expression(factor.name())
                factor_columns.append(data_mapping[symbol.name])
            ids_columns.append(interact_ids(*factor_columns))

        return np.column_stack(ids_columns)

    def _build_absorb(self, ids: Array) -> 'Absorb':
        """Build a function used to absorb fixed effects defined by columns of IDs."""
        import pyhdfe
        return Absorb(pyhdfe.create(
            ids, drop_singletons=False, compute_degrees=False, residualize_method=self._absorb_method,
            options=self._absorb_options
        ))


class Absorb(object):
    """Wrapper for PyHDFE fixed effect absorption."""

    def __init__(self, algorithm: Any) -> None:
        """Store the PyHDFE algorithm."""
        self.algorithm = algorithm

    def __call__(self, matrix: Array) -> Tuple[Array, List[Error]]:
        """Handle any absorption errors."""
        errors: List[Error] = []
        try:
            matrix = self.algorithm.residualize(matrix)
        except Exception as exception:
            errors.append(exceptions.AbsorptionError(exception))
        return matrix, errors


class ColumnFormulation(object):
    """Information about a single column in a matrix formulation."""

    names: Set[str]
    expression: sp.Expr
    derivatives: Dict[Tuple[str, int], sp.Expr]

    def __init__(self, formula: str, expression: sp.Expr) -> None:
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

        # cache evaluated derivatives
        self.derivatives: Dict[Tuple[str, int], sp.Expr] = {}

    def __str__(self) -> str:
        """Format the expression as a string."""
        return str(self.expression)

    def __repr__(self) -> str:
        """Defer to the string representation."""
        return str(self)

    def __hash__(self) -> int:
        """Hash the expression."""
        return hash(self.expression)

    def __eq__(self, other: Any) -> bool:
        """Defer to the string representation."""
        return str(self) == str(other)

    def evaluate(self, data: Mapping, data_override: Optional[Mapping] = None) -> Array:
        """Evaluate the SymPy column expression at the values supplied by mappings from variable names to arrays."""
        return evaluate_expression(self.expression, data, data_override)

    def evaluate_derivative(
            self, name: str, data: Mapping, data_override: Optional[Mapping] = None, order: int = 1) -> Array:
        """Differentiate the SymPy column expression with respect to a variable name and evaluate the derivative at
        values supplied by mappings from variable names to arrays.
        """
        return evaluate_expression(self.differentiate(name, order), data, data_override)

    def differentiate(self, name: str, order: int = 1) -> sp.Expr:
        """Differentiate the SymPy column expression with respect to a variable name. Cache calls for speed."""
        if (name, order) not in self.derivatives:
            self.derivatives[(name, order)] = self.expression
            for _ in range(order):
                self.derivatives[(name, order)] = self.derivatives[(name, order)].diff(sp.Symbol(name))

        return self.derivatives[(name, order)]


class EvaluationEnvironment(patsy.eval.EvalEnvironment):
    """Execution environment that parses SymPy expressions from strings and evaluates them at values from a data mapping
    represented as a namespace.
    """

    flags: int
    _namespaces: List[Data]

    def subset(self, _: Any) -> patsy.eval.EvalEnvironment:
        """Override the default patsy behavior to create a new patsy evaluation environment instead of a new instance
        of this same class.
        """
        return self.__class__(self._namespaces, self.flags)

    def eval(self, string: str, **_: Any) -> Array:
        """Parse a SymPy expression from a string and evaluate it at data represented as the environment's only
        namespace.
        """
        data = self._namespaces[0].copy()

        # parse the SymPy expression, preserving the function that marks variables as categorical
        expression = parse_expression(string, mark_categorical=True)

        # replace categorical variables with unicode objects and explicitly mark them as categorical so that labels are
        #   unique and so that all categorical variables are treated the same
        C = sp.Function('C')
        for symbol in expression.free_symbols:
            if not np.issubdtype(data[symbol.name].dtype, getattr(np, 'number')):
                expression = expression.replace(symbol, C(symbol))
                data[symbol.name] = data[symbol.name].astype(np.unicode_).astype(np.object_)

        # evaluate the expression and handle universally-marked categorical variables with a non-default coding class
        evaluated = evaluate_expression(expression, self._namespaces[0], function_mapping={
            'C': functools.partial(patsy.builtins.C, contrast=CategoricalTreatment)
        })

        # if the evaluated expression is a scalar, it is a constant that needs to be repeated
        if isinstance(evaluated, (numbers.Number, np.ndarray)) and np.asarray(evaluated).size == 1:
            size = next(iter(data.values())).shape[0]
            evaluated = np.ones(size) * evaluated
        return evaluated


class CategoricalTreatment(patsy.contrasts.Treatment):
    """Standard treatment coding with universal level naming and additional functionality used to parse SymPy symbols
    from categorical variable names that include levels.
    """

    @classmethod
    def code_with_intercept(cls, levels: Sequence) -> patsy.contrasts.ContrastMatrix:
        """Code a full-rank contrast matrix."""
        return patsy.contrasts.ContrastMatrix(np.eye(len(levels)), [cls.format_suffix(l) for l in levels])

    @classmethod
    def code_without_intercept(cls, levels: Sequence) -> patsy.contrasts.ContrastMatrix:
        """Code a reduced-rank contrast matrix. Choose the first level as the reference level."""
        contrast = cls.code_with_intercept(levels[1:])
        contrast.matrix = np.r_[np.zeros((1, contrast.matrix.shape[1])), contrast.matrix]
        return contrast

    @staticmethod
    def format_suffix(level: Any) -> str:
        """Format a level as an indicator variable suffix. Using the level's string representation guarantees a
        one-to-one mapping between indicator variables and their names because categorical variables have already been
        normalized as string objects. It also guarantees that the indicator variable name can be represented as SymPy
        symbol.
        """
        return f'[{level!r}]'

    @staticmethod
    def remove_suffix(name: str) -> Optional[str]:
        """Drop the suffix from an indicator variable name."""
        return name[:name.index('[')] if '[' in name else None

    @staticmethod
    def extract_suffix(name: str) -> Optional[str]:
        """Keep only the suffix from an indicator variable name."""
        return name[name.index('['):] if '[' in name else None

    @classmethod
    def parse_base_symbol(cls, name: str) -> Optional[sp.Symbol]:
        """Parse a SymPy symbol from an indicator variable name that represents the original categorical variable. The
        base name needs to be parsed because it could be encapsulated by the function that explicitly marks categorical
        variables.
        """
        base = cls.remove_suffix(name)
        return None if base is None else parse_expression(base)

    @classmethod
    def parse_full_symbol(cls, name: str) -> Optional[sp.Symbol]:
        """Parse a SymPy symbol from an indicator variable name that includes the suffix and hence represents the actual
        indicator variable.
        """
        suffix = cls.extract_suffix(name)
        return None if suffix is None else sp.Symbol(f'{cls.parse_base_symbol(name)}{suffix}')


def parse_terms(formula: str) -> List[patsy.desc.Term]:
    """Parse patsy terms from a string. Validate that the string contains only right-hand side terms."""
    description = patsy.highlevel.ModelDesc.from_formula(formula)
    if description.lhs_termlist:
        end = formula.index('~') + 1 if '~' in formula else len(formula)
        raise patsy.PatsyError("Formulas should not have left-hand sides.", patsy.origin.Origin(formula, 0, end))
    return description.rhs_termlist


def design_matrix(terms: Sequence[patsy.desc.Term], data: Mapping) -> patsy.design_info.DesignInfo:
    """Design a patsy matrix."""
    return patsy.build.design_matrix_builders([terms], lambda: iter([data]), EvaluationEnvironment([data]))[0]


def build_matrix(design: patsy.design_info.DesignInfo, data: Mapping) -> Array:
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


def parse_term_expression(term: patsy.desc.Term) -> sp.Expr:
    """Multiply the SymPy expressions parsed from each factor in a patsy term."""
    expression = sp.Integer(1)
    for factor in term.factors:
        try:
            expression *= parse_expression(factor.name())
        except Exception as exception:
            message = "Failed to parse a term because of the above exception."
            raise patsy.PatsyError(message, factor.origin) from exception

    return expression


def parse_expression(string: str, mark_categorical: bool = False) -> sp.Expr:
    """Parse a SymPy expression from a string. Optionally, preserve the categorical marker function instead of treating
    it like the identify function.
    """

    # list reserved patsy and SymPy names that represent special functions and classes
    patsy_function_names = {'I', 'C'}
    sympy_function_names = {'log', 'exp'}
    sympy_class_names = {'Add', 'Mul', 'Pow', 'Integer', 'Float', 'Symbol'}

    # build a mapping from reserved names to the functions and classes that they represent (patsy functions are dealt
    #   with after parsing)
    mapping = {n: sp.Function(n) for n in patsy_function_names}
    mapping.update({n: getattr(sp, n) for n in sympy_function_names | sympy_class_names})

    def transform_tokens(tokens: List[Tuple[int, str]], _: Any, __: Any) -> List[Tuple[int, str]]:
        """Validate a list of tokens and add any unrecognized names as new SymPy symbols."""
        transformed: List[Tuple[int, str]] = []
        symbol_candidate = None
        for code, value in tokens:
            if code not in {token.NAME, token.OP, token.NUMBER, token.NEWLINE, token.ENDMARKER}:
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

    # define a function that validates the appearance of categorical marker functions
    def validate_categorical(candidate: sp.Expr, depth: int = 0, categorical: bool = False) -> None:
        """Recursively validate that all categorical marker functions in an expression accept only a single variable
        argument and that they are not arguments to other functions.
        """
        if categorical and depth > 1:
            raise ValueError("The C function must not be an argument to another function.")
        for arg in candidate.args:
            if categorical and not isinstance(arg, sp.Symbol):
                raise ValueError("The C function accepts only a single variable.")
            validate_categorical(arg, depth + 1, candidate.func == mapping['C'])

    # parse the expression, validate it by attempting to represent it as a string, and validate categorical markers
    try:
        expression = sympy.parsing.sympy_parser.parse_expr(string, mapping, [transform_tokens], evaluate=False)
        str(expression)
        validate_categorical(expression)
    except (TypeError, ValueError) as exception:
        raise ValueError(f"The expression '{string}' is malformed because of the above exception.") from exception

    # replace patsy functions with the identity function, unless categorical variables are to be explicitly marked
    for name in patsy_function_names:
        if name != 'C' or not mark_categorical:
            expression = expression.replace(mapping[name], sp.Id)

    return expression


def evaluate_expression(
        expression: Union[sp.Expr, sp.Symbol], data: Mapping, data_override: Optional[Mapping] = None,
        function_mapping: Optional[Mapping[str, Callable]] = None) -> Array:
    """Evaluate a SymPy expression at data mapping variable names to arrays. Optionally, supplement the default suite of
    NumPy functions with a mapping from non-default function names to functions.
    """
    if expression.is_number:
        return np.asarray(float(expression), options.dtype)
    if expression.is_symbol:
        return get_symbol_data(expression, data, data_override)
    symbols = list(expression.free_symbols)
    modules = [function_mapping or {}, 'numpy']
    columns = (get_symbol_data(s, data, data_override) for s in symbols)
    return sp.lambdify(symbols, expression, modules)(*columns)


def get_symbol_data(symbol: sp.Symbol, data: Mapping, data_override: Optional[Mapping] = None) -> Any:
    """Fetch data corresponding to a symbol from data mapping variable names to arrays."""
    try:
        assert data_override is not None
        return data_override[symbol.name]
    except Exception:
        return data[symbol.name]

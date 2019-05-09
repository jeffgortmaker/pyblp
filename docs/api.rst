.. currentmodule:: pyblp


API Documentation
=================

The majority of the package consists of classes, which compartmentalize different aspects of the BLP model. There are some convenience functions as well.


Configuration Classes
---------------------

Various components of the package require configurations for how to approximate integrals, solve fixed point problems, and solve optimimzation problems. Such configurations are specified with the following classes.

.. autosummary::
   :toctree: _api
   :template: class_without_methods.rst

   Formulation
   Integration
   Iteration
   Optimization


Data Construction Functions
---------------------------

There are also a number of convenience functions that can be used to construct common components of product and agent data.

.. autosummary::
   :toctree: _api

   build_matrix
   build_blp_instruments
   build_differentiation_instruments
   build_id_data
   build_ownership
   build_integration


Problem Class
-------------

Given data and appropriate configurations, a BLP-type problem can be structured by initializing the following class.

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

   Problem

Once initialized, the following method solves the problem.

.. autosummary::
   :toctree: _api

   Problem.solve


Problem Results Class
---------------------

Solved problems return the following results class.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   ProblemResults

In addition to class attributes, other post-estimation outputs can be estimated with the following methods, which each return an array.

.. autosummary::
   :toctree: _api

   ProblemResults.compute_aggregate_elasticities
   ProblemResults.compute_elasticities
   ProblemResults.compute_diversion_ratios
   ProblemResults.compute_long_run_diversion_ratios
   ProblemResults.extract_diagonals
   ProblemResults.extract_diagonal_means
   ProblemResults.compute_costs
   ProblemResults.compute_approximate_prices
   ProblemResults.compute_prices
   ProblemResults.compute_shares
   ProblemResults.compute_hhi
   ProblemResults.compute_markups
   ProblemResults.compute_profits
   ProblemResults.compute_consumer_surpluses

A parametric bootstrap can be used, for example, to compute standard errors for the above post-estimation outputs. The following method returns a results class with all of the above methods, which returns a distribution of post-estimation outputs corresponding to different bootstrapped samples.

.. autosummary::
   :toctree: _api

   ProblemResults.bootstrap

Optimal instruments, which also return a results class instead of an array, can be estimated with the following method.

.. autosummary::
   :toctree: _api

   ProblemResults.compute_optimal_instruments


Boostrapped Problem Results Class
---------------------------------

Parametric bootstrap computation returns the following class.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_methods_or_signature.rst

   BootstrappedResults

This class has all of the same methods as :class:`ProblemResults`, except for :meth:`ProblemResults.bootstrap` and :meth:`ProblemResults.compute_optimal_instruments`.


Optimal Instrument Results Class
--------------------------------

Optimal instrument computation returns the following results class.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   OptimalInstrumentResults

The optimal instrument results can be converted into a :class:`Problem` with the following method.

.. autosummary::
   :toctree: _api

   OptimalInstrumentResults.to_problem

This method returns the following class, which behaves exactly like a :class:`Problem`.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_methods_or_signature.rst

   OptimalInstrumentProblem


Micro Moment Classes
--------------------

Micro moment configurations can be passed to :meth:`Problem.solve`. Only one type of micro moment is currently supported.

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

   ProductsAgentsCovarianceMoment


Simulation Class
----------------

In addition to reading from data files, data can be simulated by initializing the following class.

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

   Simulation

Once initialized, the following method computes equilibrium prices and shares.

.. autosummary::
   :toctree: _api

   Simulation.solve


Simulation Results Class
------------------------

Solved simulations return the following results class.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   SimulationResults

The simulation results can be converted into a :class:`Problem` with the following method.

.. autosummary::
   :toctree: _api

   SimulationResults.to_problem

Simulation results can also be used to compute micro moment values.

.. autosummary::
   :toctree: _api

   SimulationResults.compute_micro


Structured Data Classes
-----------------------

Product and agent data that are passed or constructed by :class:`Problem` and :class:`Simulation` are structured internally into classes with field names that more closely resemble BLP notation. Although these structured data classes are not directly constructable, they can be accessed with :class:`Problem` and :class:`Simulation` class attributes. It can be helpful to compare these structured data classes with the data or configurations used to create them.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_methods_or_signature.rst

   Products
   Agents


Multiprocessing
---------------

A context manager can be used to enable parallel processing for methods that perform market-by-market computation.

.. autosummary::
   :toctree: _api

   parallel


Options and Example Data
------------------------

In addition to classes and functions, there are also two modules that can be used to configure global package options and locate example data that comes with the package.

.. autosummary::
   :toctree: _api

   options
   data


Exceptions
----------

When errors occur, they will either be displayed as warnings or raised as exceptions.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   exceptions.MultipleErrors
   exceptions.NonpositiveCostsError
   exceptions.InvalidParameterCovariancesError
   exceptions.InvalidMomentCovariancesError
   exceptions.DeltaFloatingPointError
   exceptions.CostsFloatingPointError
   exceptions.MicroMomentsFloatingPointError
   exceptions.XiByThetaJacobianFloatingPointError
   exceptions.OmegaByThetaJacobianFloatingPointError
   exceptions.MicroMomentsByThetaJacobianFloatingPointError
   exceptions.SyntheticPricesFloatingPointError
   exceptions.SyntheticSharesFloatingPointError
   exceptions.SyntheticMicroMomentsFloatingPointError
   exceptions.EquilibriumRealizationFloatingPointError
   exceptions.PostEstimationFloatingPointError
   exceptions.AbsorptionConvergenceError
   exceptions.ThetaConvergenceError
   exceptions.DeltaConvergenceError
   exceptions.SyntheticPricesConvergenceError
   exceptions.EquilibriumPricesConvergenceError
   exceptions.ObjectiveReversionError
   exceptions.GradientReversionError
   exceptions.DeltaReversionError
   exceptions.CostsReversionError
   exceptions.MicroMomentsReversionError
   exceptions.XiByThetaJacobianReversionError
   exceptions.OmegaByThetaJacobianReversionError
   exceptions.MicroMomentsByThetaJacobianReversionError
   exceptions.AbsorptionInversionError
   exceptions.HessianEigenvaluesError
   exceptions.FittedValuesInversionError
   exceptions.SharesByXiJacobianInversionError
   exceptions.IntraFirmJacobianInversionError
   exceptions.LinearParameterCovariancesInversionError
   exceptions.GMMParameterCovariancesInversionError
   exceptions.GMMMomentCovariancesInversionError

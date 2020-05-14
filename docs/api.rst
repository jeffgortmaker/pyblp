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
   data_to_dict


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


Micro Moment Classes
--------------------

Micro moment configurations can be passed to :meth:`Problem.solve`.

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

   DemographicExpectationMoment
   DemographicCovarianceMoment
   DiversionProbabilityMoment
   DiversionCovarianceMoment


Problem Results Class
---------------------

Solved problems return the following results class.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   ProblemResults

The results can be converted into a dictionary.

.. autosummary::
   :toctree: _api

   ProblemResults.to_dict

The following methods test the validity of overidentifying and model restrictions.

.. autosummary::
   :toctree: _api

   ProblemResults.run_hansen_test
   ProblemResults.run_distance_test
   ProblemResults.run_lm_test
   ProblemResults.run_wald_test

In addition to class attributes, other post-estimation outputs can be estimated market-by-market with the following methods, which each return an array.

.. autosummary::
   :toctree: _api

   ProblemResults.compute_aggregate_elasticities
   ProblemResults.compute_elasticities
   ProblemResults.compute_diversion_ratios
   ProblemResults.compute_long_run_diversion_ratios
   ProblemResults.compute_probabilities
   ProblemResults.extract_diagonals
   ProblemResults.extract_diagonal_means
   ProblemResults.compute_delta
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

Importance sampling can be used to create new integration nodes and weights. Its method also returns a results class.

.. autosummary::
   :toctree: _api

   ProblemResults.importance_sampling


Bootstrapped Problem Results Class
---------------------------------

Parametric bootstrap computation returns the following class.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_methods_or_signature.rst

   BootstrappedResults

This class has all of the same methods as :class:`ProblemResults`, except for :meth:`ProblemResults.bootstrap`, :meth:`ProblemResults.compute_optimal_instruments`, and :meth:`ProblemResults.importance_sampling`. It can also be converted into a dictionary.

.. autosummary::
   :toctree: _api

   BootstrappedResults.to_dict


Optimal Instrument Results Class
--------------------------------

Optimal instrument computation returns the following results class.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   OptimalInstrumentResults

The results can be converted into a dictionary.

.. autosummary::
   :toctree: _api

   OptimalInstrumentResults.to_dict

They can also be converted into a :class:`Problem` with the following method.

.. autosummary::
   :toctree: _api

   OptimalInstrumentResults.to_problem

This method returns the following class, which behaves exactly like a :class:`Problem`.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_methods_or_signature.rst

   OptimalInstrumentProblem


Importance Sampling Results Class
---------------------------------

Importance sampling returns the following results class:

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   ImportanceSamplingResults

The results can be converted into a dictionary.

.. autosummary::
   :toctree: _api

   ImportanceSamplingResults.to_dict

They can also be converted into a :class:`Problem` with the following method.

.. autosummary::
   :toctree: _api

   ImportanceSamplingResults.to_problem

This method returns the following class, which behaves exactly like a :class:`Problem`.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_methods_or_signature.rst

   ImportanceSamplingProblem


Simulation Class
----------------

The following class allows for evaluation of more complicated counterfactuals than is possible with :class:`ProblemResults` methods, or for simulation of synthetic data from scratch.

.. autosummary::
   :toctree: _api
   :template: class_with_signature.rst

   Simulation

Once initialized, the following method replaces prices and shares with equilibrium values that are consistent with true parameters.

.. autosummary::
   :toctree: _api

   Simulation.replace_endogenous

A less common way to solve the simulation is to assume simulated prices and shares represent and equilibrium and to replace exogenous variables instead.

.. autosummary::
   :toctree: _api

   Simulation.replace_exogenous


Simulation Results Class
------------------------

Solved simulations return the following results class.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   SimulationResults

The results can be converted into a dictionary.

.. autosummary::
   :toctree: _api

   SimulationResults.to_dict

They can also be converted into a :class:`Problem` with the following method.

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
   exceptions.NonpositiveSyntheticCostsError
   exceptions.InvalidParameterCovariancesError
   exceptions.InvalidMomentCovariancesError
   exceptions.DeltaNumericalError
   exceptions.CostsNumericalError
   exceptions.MicroMomentsNumericalError
   exceptions.XiByThetaJacobianNumericalError
   exceptions.OmegaByThetaJacobianNumericalError
   exceptions.MicroMomentsByThetaJacobianNumericalError
   exceptions.MicroMomentCovariancesNumericalError
   exceptions.SyntheticPricesNumericalError
   exceptions.SyntheticSharesNumericalError
   exceptions.SyntheticDeltaNumericalError
   exceptions.SyntheticCostsNumericalError
   exceptions.SyntheticMicroMomentsNumericalError
   exceptions.EquilibriumRealizationNumericalError
   exceptions.PostEstimationNumericalError
   exceptions.AbsorptionError
   exceptions.ClippedSharesError
   exceptions.ThetaConvergenceError
   exceptions.DeltaConvergenceError
   exceptions.SyntheticPricesConvergenceError
   exceptions.SyntheticDeltaConvergenceError
   exceptions.EquilibriumPricesConvergenceError
   exceptions.ObjectiveReversionError
   exceptions.GradientReversionError
   exceptions.DeltaReversionError
   exceptions.CostsReversionError
   exceptions.MicroMomentsReversionError
   exceptions.XiByThetaJacobianReversionError
   exceptions.OmegaByThetaJacobianReversionError
   exceptions.MicroMomentsByThetaJacobianReversionError
   exceptions.HessianEigenvaluesError
   exceptions.FittedValuesInversionError
   exceptions.SharesByXiJacobianInversionError
   exceptions.IntraFirmJacobianInversionError
   exceptions.LinearParameterCovariancesInversionError
   exceptions.GMMParameterCovariancesInversionError
   exceptions.GMMMomentCovariancesInversionError

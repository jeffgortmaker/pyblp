.. currentmodule:: pyblp


API Documentation
=================

The majority of the package consists of classes, which compartmentalize different aspects of the BLP model. There are some convenience functions as well.

Many of the pages that the below summaries link to contain example code. For the sake of brevity, the following lines of code, which import the package, import :mod:`numpy`, and limit verbosity, are executed before each example.

.. literalinclude:: include.py
  :language: ipython


Configuration Classes
---------------------

Various components of the package require configurations for how to approximate integrals, solve fixed point problems, and solve optimimzation problems. Such configurations are specified with the following classes.

.. autosummary::
   :toctree: api
   :template: class_without_methods.rst

   Integration
   Iteration
   Optimization


Data Construction Functions
---------------------------

There are also a number of convenience functions that can be used to construct common parts of product data.

.. autosummary::
   :toctree: api

   build_id_data
   build_indicators
   build_ownership
   build_blp_instruments


Simulation Class
----------------

In addition to reading from data files, product data can be simulated by initializing the following class.

.. autosummary::
   :toctree: api
   :template: class_with_signature.rst

   Simulation

Once initialized, the following method computes Bertrand-Nash prices and shares.

.. autosummary::
   :toctree: api

   Simulation.solve


Problem Class
-------------

Given real or simulated data and appropriate configurations, the BLP problem can be structured by initializing the following class.

.. autosummary::
   :toctree: api
   :template: class_with_signature.rst

   Problem

Once initialized, the following method solves the problem.

.. autosummary::
   :toctree: api

   Problem.solve


Problem Results Class
---------------------

Solved BLP problems return the following results class.

.. autosummary::
   :nosignatures:
   :toctree: api
   :template: class_without_signature.rst

   Results

In addition to class attributes, other post-estimation outputs can be computed with the following methods.

.. autosummary::
   :toctree: api

   Results.compute_aggregate_elasticities
   Results.compute_aggregate_price_elasticities
   Results.compute_elasticities
   Results.compute_price_elasticities
   Results.compute_diversion_ratios
   Results.compute_price_diversion_ratios
   Results.compute_long_run_diversion_ratios
   Results.extract_diagonals
   Results.extract_diagonal_means
   Results.compute_costs
   Results.solve_approximate_merger
   Results.solve_merger
   Results.compute_shares
   Results.compute_hhi
   Results.compute_markups
   Results.compute_profits
   Results.compute_consumer_surpluses


Structured Data Classes
-----------------------

Product and agent data that are passed or constructed by :class:`Problem` and :class:`Simulation` are stuctured internally into classes with field names that more closely resemble BLP notation. Although these structured data classes are not directly constructable, they can be accessed with :class:`Problem` and :class:`Simulation` class attributes. It can be helpful to compare these structured data classes with the data or configurations used to create them.

.. autosummary::
   :nosignatures:
   :toctree: api
   :template: class_without_methods.rst

   primitives.Products
   primitives.Agents


Options and Example Data
------------------------

In addition to classes and functions, there are also two modules that can be used to configure global package options and locate example data that comes with the package.

.. autosummary::
   :toctree: api

   options
   data

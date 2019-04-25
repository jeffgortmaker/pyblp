{% set path = env.doc2path(env.docname, base=None).replace('\\', '/') %}

.. only:: html

   .. nbinfo::

      Download the `Jupyter Notebook <https://jupyter.org/>`_ for this section: :download:`{{ path.rsplit('/', 1)[1] }} </{{ path.replace('_notebooks', '_downloads') }}>`

.. raw:: latex

   \begin{landscape}

.. only:: latex

   .. nbinfo::

      The :rtd:`online version <{{ path.replace('ipynb', 'html') }}>` of the following section may be easier to read.

{% set path = env.doc2path(env.docname, base=None).replace('\\', '/') %}

.. nbinfo::

   Download the `Jupyter Notebook <http://jupyter.org/>`_ for this section: :download:`{{ path.rsplit('/', 1)[1] }} </{{ path.replace('_notebooks', '_downloads') }}>`

Installation
============
The most recent release can be installed from
`PyPI <https://pypi.org/project/gptchem>`_ with:

.. code-block:: shell

    $ pip install gptchem

The most recent code and data can be installed directly from GitHub with:

.. code-block:: shell

    $ pip install git+https://github.com/kjappelbaum/gptchem.git

To install in development mode, use the following:

.. code-block:: shell

    $ git clone git+https://github.com/kjappelbaum/gptchem.git
    $ cd gptchem
    $ pip install -e .


Extras 
--------
If you want to reproduce the experiments, you will need to install some additional dependencies (which can be done with ``pip install gptchem[experiments]``). Some experiments, however, might need additional dependencies that cannot easily be installed with pip. In this case, you will need to install the dependencies manually. Please refer to the respective experiment for more information.
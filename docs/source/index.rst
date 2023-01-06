gptchem |release| Documentation
===============================

Cookiecutter
------------
This package was created with the `cookiecutter <https://github.com/cookiecutter/cookiecutter>`_
package using `cookiecutter-snekpack <https://github.com/cthoyt/cookiecutter-snekpack>`_ template.
It comes with the following:

- Standard `src/` layout
- Declarative setup with `setup.cfg` and `pyproject.toml`
- Reproducible tests with `pytest` and `tox`
- A command line interface with `click`
- A vanity CLI via python entrypoints
- Version management with `bumpversion`
- Documentation build with `sphinx`
- Testing of code quality with `flake8` in `tox`
- Testing of documentation coverage with `docstr-coverage` in `tox`
- Testing of documentation format and build in `tox`
- Testing of package metadata completeness with `pyroma` in `tox`
- Testing of MANIFEST correctness with `check-manifest` in `tox`
- Testing of optional static typing with `mypy` in `tox`
- A `py.typed` file so other packages can use your type hints
- Automated running of tests on each push with GitHub Actions
- Configuration for `ReadTheDocs <https://readthedocs.org/>`_
- A good base `.gitignore` generated from `gitignore.io <https://gitignore.io>`_.
- A pre-formatted README with badges
- A pre-formatted LICENSE file with the MIT License (you can change this to whatever you want, though)
- A pre-formatted CONTRIBUTING guide
- Automatic tool for releasing to PyPI with ``tox -e finish``
- A copy of the `Contributor Covenant <https://www.contributor-covenant.org>`_ as a basic code of conduct

Table of Contents
-----------------
.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: start

   installation

Indices and Tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

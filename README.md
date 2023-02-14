
<p align="center">
  <img src="https://github.com/kjappelbaum/gptchem/raw/main/docs/source/static/grid_0.png" height="150">
</p>


<h1 align="center">
  gptchem
</h1>

<p align="center">
    <a href="https://github.com/kjappelbaum/gptchem/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/kjappelbaum/gptchem/workflows/Tests/badge.svg" />
    </a>
    <!-- <a href="https://pypi.org/project/gptchem">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/gptchem" />
    </a>
    <a href="https://pypi.org/project/gptchem">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/gptchem" />
    </a> -->
    <a href="https://github.com/kjappelbaum/gptchem/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/gptchem" />
    </a>
    <a href='https://gptchem.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/gptchem/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/kjappelbaum/gptchem/branch/main">
        <img src="https://codecov.io/gh/kjappelbaum/gptchem/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com/kjappelbaum/gptchem/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

Use GPT-3 to solve chemistry problems.
Most of the repo is currently not intended for use as library but as documentation of our experiments. 
We'll factor out the experiments (that come with tricky dependencies) into its own repository over time.

## 💪 Getting Started

```python 
from gptchem.gpt_classifier import GPTClassifier 
from gptchem.tuner import Tuner 

classifier = GPTClassifier(
    property_name="transition wavelength", # this is the property name we will use in the prompt template
    tuner=Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False),
)

classifier.fit(["CC", "CDDFSS"], [0, 1])
predictions = classifier.predict(['CCCC', 'CCCCCCCC'])
```

## 🚀 Installation

<!-- Uncomment this section after your first ``tox -e finish``
The most recent release can be installed from
[PyPI](https://pypi.org/project/gptchem/) with:

```bash
$ pip install gptchem
```
-->

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/kjappelbaum/gptchem.git
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/kjappelbaum/gptchem/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.


### 📖 Citation

Citation goes here!


<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

-->

<!--
### 💰 Funding

This project has been supported by the following grants:

| Funding Body | Program                                                                                                                       | Grant         |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------- | ------------- |
| DARPA        | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009 |
-->

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>


The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/kjappelbaum/gptchem.git
$ cd gptchem
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/kjappelbaum/gptchem/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/kjappelbaum/gptchem.git
$ cd gptchem
$ tox -e docs
$ open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/gptchem/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.
</details>

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

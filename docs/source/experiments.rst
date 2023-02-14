Experiments
===============

For reproducing experiments from our paper, please have a look at the ``experiments`` folder.

You can find prior experiments in `our old repository <https://github.com/kjappelbaum/gpt3forchem>`_.

0. No fine-tuning
-------------------

The ``00_no_fine_tuning`` folder contains the code for the experiments we conducted to investigate the performance of ``GPT`` without any fine-tuning.
We basically see that without fine-tuning ``GPT`` cannot answer our questions---also simply because it does not know the format of the prompts and the format of the completions we would expect.


1. Tuning parameter influence 
------------------------------

The ``01_tuning_parameter_influence`` folder contains the code for the experiments we conducted to investigate the influence of a couple of tuning parameters on the performance of the classification performance. 
For computational reasons, we limited these experiments to a binary classification on the photoswitch dataset.
For this experiments we split the dataset into two balanced classes. Therefore, a dummy classifier would have an accuracy of 50%.

.. note:: 

    We recommend to always also train dummy models, e.g. :py:class:`sklearn.dummy.DummyClassifier` or :py:class:`sklearn.dummy.DummyRegressor` to get a sense of the performance of a random model.
    Only consider GPT-3's predictions if they are better than the dummy model.

We investigated the following parameters:

- ``num_train_epochs``: Number of training epochs
- ``learning_rate_multiplier``: Multiplier for the learning rate
- ``base_model``: The base model used for fine-tuning
    - GPT-3:
        - ``text-davinci-003``
        - ``text-ada-001``
        - ``text-babbage-002``
        - ``text-curie-002``
    At the beginning of Jan 2023, Codex models could not yet be fine-tuned.

To run the baselines, the following additional dependencies are needed:

- ``tabpfn``. Follow the installation instructions of the `tabpfn repository <https://github.com/automl/TabPFN>`_.
- ``gpflow``. Follow the installation instructions of the `photoswitch dataset repository <https://github.com/Ryan-Rhys/The-Photoswitch-Dataset>`_


3. Classification experiments
-------------------------------

Each folder will contain a Python script to run the experiments and a notebook to analyze the results.

You will find subfolders for the following experiments: 

HOMO/LUMO gaps
..................

See the folder `bandgap`

To run the baselines, the following additional dependencies are needed:

- ``molclr``. Follow the installation instructions of `our fork of the MolCLR repository <https://github.com/kjappelbaum/MolCLR>`_.
- ``gpflow``. Follow the installation instructions of the `photoswitch dataset repository <https://github.com/Ryan-Rhys/The-Photoswitch-Dataset>`_
- ``optuna``  Follow the installation instructions of the `optuna repository <https://github.com/optuna/optuna>`_.



Heat capacity 
...............

See the folder `cv`

To run the baseline, follow the `instructed provided by Moosavi et al. <https://github.com/SeyedMohamadMoosavi/tools-cp-porousmat>`_. For the composition-based baseline, you also need CrabNet. Follow the installation instructions on Sterling Baird's `fork of the CrabNet repository <https://github.com/sparks-baird/CrabNet>`_.


High entropy alloy phase
..............................

See the folder `hea_phase`


High entropy alloy single vs multiphase
.........................................

See the folder `hea_single_vs_multiphase`

To run the baselines, the following additional dependencies are needed:

- `automatminer` Follow the installation instructions of the `automatminer repository <https://github.com/hackingmaterials/automatminer>`_.
- `CrabNet`  Follow the installation instructions on Sterling Baird's `fork of the CrabNet repository <https://github.com/sparks-baird/CrabNet>`_.


Henry coefficients
......................  

See the folder `henry`. 

To run the baselines, the following additional dependencies are needed:

- ``optuna``  Follow the installation instructions of the `optuna repository <https://github.com/optuna/optuna>`_.



Lipophilicity
.................

See the folder `lipophilicity`

To run the baselines, the following additional dependencies are needed:

- ``molclr``. Follow the installation instructions of `our fork of the MolCLR repository <https://github.com/kjappelbaum/MolCLR>`_.
- ``gpflow``. Follow the installation instructions of the `photoswitch dataset repository <https://github.com/Ryan-Rhys/The-Photoswitch-Dataset>`_
- ``optuna``  Follow the installation instructions of the `optuna repository <https://github.com/optuna/optuna>`_.


Matbench
...............

See the folder `matbench`. In there, there is one folder per task. 

- `automatminer` Follow the installation instructions of the `automatminer repository <https://github.com/hackingmaterials/automatminer>`_.
- `MODNet` Follow the installation instructions of the `MODNet repository <https://github.com/ppdebreuck/modnet>`_.\




OPV
.......

See the folder `opv`

To run the baselines, the following additional dependencies are needed:

- ``molclr``. Follow the installation instructions of `our fork of the MolCLR repository <https://github.com/kjappelbaum/MolCLR>`_.
- ``gpflow``. Follow the installation instructions of the `photoswitch dataset repository <https://github.com/Ryan-Rhys/The-Photoswitch-Dataset>`_
- ``optuna``  Follow the installation instructions of the `optuna repository <https://github.com/optuna/optuna>`_.


Photoswitches
...............

See the folder `photoswitch`

To run the baselines, the following additional dependencies are needed:

- ``molclr``. Follow the installation instructions of `our fork of the MolCLR repository <https://github.com/kjappelbaum/MolCLR>`_.
- ``gpflow``. Follow the installation instructions of the `photoswitch dataset repository <https://github.com/Ryan-Rhys/The-Photoswitch-Dataset>`_
- ``optuna``  Follow the installation instructions of the `optuna repository <https://github.com/optuna/optuna>`_.



Polymers 
...............

To run the baselines, the following additional dependencies are needed: 

- ``optuna``  Follow the installation instructions of the `optuna repository <https://github.com/optuna/optuna>`_.


C-N cross-coupling
......................

See the folder `rxn_doyle`.

To run the baselines, the following additional dependencies are needed: 

- ``gauche``. Follow the installation instructions of the `gauche repository <https://github.com/leojklarner/gauche/>`_.`


C-C cross-coupling
......................

See the folder `rxn_suzuki`.

To run the baselines, the following additional dependencies are needed: 

- ``gauche``. Follow the installation instructions of the `gauche repository <https://github.com/leojklarner/gauche/>`_.`


Solubility
...............

See the folder `solubility`

To run the baselines, the following additional dependencies are needed:

- ``gauche``. Follow the installation instructions of the `gauche repository <https://github.com/leojklarner/gauche/>`_.`
- ``molclr``. Follow the installation instructions of `our fork of the MolCLR repository <https://github.com/kjappelbaum/MolCLR>`_.
- ``gpflow``. Follow the installation instructions of the `photoswitch dataset repository <https://github.com/Ryan-Rhys/The-Photoswitch-Dataset>`_
- ``optuna``  Follow the installation instructions of the `optuna repository <https://github.com/optuna/optuna>`_.
- ``deepchem`` Follow the installation instructions of the `deepchem repository <https://deepchem.io/>`_.

4. Regression experiments
----------------------------

The regression experiments follow the same structure as the classification experiments.

5. Inverse design 
---------------------

The evaluation of the HOMO-LUMO gap inverse design expects that the scripts are run on a server with `slurm` as job scheduler. 

Some experiments hard-code pretrained models. You won't have access to those as they are limited to our organization. However, you can fine-tune a model yourself and then change the modelname.

6. Prompt structure 
----------------------

In this experiments, we attempted to add some examples (with dummy data) to the prompt with the hope that this would make the learning more efficient---because the model might "in-context-learn" the structure of the prompt.
However, we found this to rather confuse the model.

7. Few shot learning
-------------------------

This contains our experiments for in-context learning. Note that this also calls the largest models and hence can be quite expensive to run.

8. Permutation test
-----------------------

Our experiment on the photoswitch where we permute the labels to see if this is different to learning from chemically meaningful data.

9. Invalid prompts
----------------------

In those experiments we used some pre-trained models. You won't have access to those as they are limited to our organization. However, you can fine-tune a model yourself and then change the modelname.

10. Functional groups 
---------------------------

In those experiments we used some pre-trained models. You won't have access to those as they are limited to our organization. However, you can fine-tune a model yourself and then change the modelname.

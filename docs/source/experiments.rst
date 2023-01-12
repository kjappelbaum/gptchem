Experiments
===============

For reproducing experiments from our paper, please have a look at the ``experiments`` folder. 

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

2. Henry coefficients
-----------------------

To ensure we can compare with well-tuned baselines, we reuse the data reported by [MoosaviDiversity]_. We also use the model with the hyperparameters reported in this work as baselines.

To run the baselines, the following additional dependencies are needed: 
 
- ``xgboost``

3. Classification experiments
-------------------------------

You will find subfolders for the following experiments: 

Polymers 
...............


To run the baselines, the following additional dependencies are needed: 
 
- ``xgboost``
- ``optuna`` 


MOFs
...........



HOMO/LUMO gaps
..................

See the folder `bandgap`

Heat capacity 
,,,,,,,,,,,,,,,,,

To run the baseline, follow the `instructed provided by Moosavi et al. <https://github.com/SeyedMohamadMoosavi/tools-cp-porousmat>`_.


Photoswitches
...............

To run the baselines, the following additional dependencies are needed: 

- ``tabpfn``. Follow the installation instructions of the `tabpfn repository <https://github.com/automl/TabPFN>`_.
- ``molclr``. Follow the installation instructions of `our fork of the MolCLR repository <https://github.com/kjappelbaum/MolCLR>`_.
- ``gpflow``. Follow the installation instructions of the `photoswitch dataset repository <https://github.com/Ryan-Rhys/The-Photoswitch-Dataset>`_

OPV
.......

To run the baselines, the following additional dependencies are needed: 

- ``tabpfn``. Follow the installation instructions of the `tabpfn repository <https://github.com/automl/TabPFN>`_.
- ``xgboost``
- ``gpflow``. Follow the installation instructions of the `photoswitch dataset repository <https://github.com/Ryan-Rhys/The-Photoswitch-Dataset>`_
- ``optuna`` 

High entropy alloys
......................

- ``hea_phase``: Classification of high-entropy alloys (HEA) into "single phase" and "multi phase" based on the dataset reported by [Pei]_. They reported an accuracy of 97% based on 10-fold cross-validation with the 1252 datapoints. They didn't report learnding curves.
- ``hea_single_vs_multiphase``: Classification of high-entropy alloys (HEA) in "fcc", "bcc", "hcp" and "multi phase" based on the dataset reported by [Pei]_. 


4. Regression experiments
----------------------------


5. Inverse design 
---------------------


High-entropy alloys 
......................

To compute the diversity metrics, the following additional dependencies are needed: 

- ``alloy2vec``. Follow the installation instructions of the `alloy2vec repository <https://github.com/peizong/alloy2vec>`_
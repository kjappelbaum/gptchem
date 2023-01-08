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

2. Henry coefficients
-----------------------

To ensure we can compare with well-tuned baselines, we reuse the data reported by [MoosaviDiversity]_. We also use the model with the hyperparameters reported in this work as baselines.

To run the baselines, the following additional dependencies are needed: 
 
- ``xgboost``

3. Classification experiments
-------------------------------

To run the baselines, the following additional dependencies are needed: 

- ``tabpfn``. Follow the installation instructions of the `tabpfn repository <https://github.com/automl/TabPFN>`_.
- ``molclr``. Follow the installation instructions of `our fork of the MolCLR repository <https://github.com/kjappelbaum/MolCLR>`_.


4. Regression experiments
----------------------------
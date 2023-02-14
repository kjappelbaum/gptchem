Getting Started
==================

For all examples, we assume that you have an openai API key that you exported as an environment variable:

.. code-block:: bash

    export OPENAI_API_KEY=sk-...    


High-level wrapper 
-----------------------

The easiest way to get started with ``gpthem`` is to use the :py:class:`gptchem.gpt_classifier.GPTClassifier` and :py:class:`gptchem.gpt_regressor.GPTRegressor` classes.
Those classes mimic the API of ``scikit-learn`` and can often be used drop-in replacements for ``scikit-learn`` models. However, in contrast to typical ``scikit-learn`` models, the ``X`` input is not an array of features, but a list of text representations of the samples.


.. code-block:: python

    from gptchem.gpt_classifier import GPTClassifier 
    from gptchem.tuner import Tuner 

    classifier = GPTClassifier(
        property_name="transition wavelength", # this is the property name we will use in the prompt template
        tuner=Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False),
    )

    classifier.fit(["CC", "CDDFSS"], [0, 1])
    predictions = classifier.predict(['CCCC', 'CCCCCCCC'])

or 

.. code-block:: python

    from gptchem.gpt_regressor import GPTRegressor 
    from gptchem.tuner import Tuner 

    regressor = GPTRegressor(
        property_name="transition wavelength", # this is the property name we will use in the prompt template
        tuner=Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False),
    )

    regressor.fit(["CC", "CDDFSS"], [0.1, 1.1])
    predictions = regressor.predict(['CCCC', 'CCCCCCCC'])

Fine-tuning
--------------

Usually, you will not get any useful predictions from the model without finetuning it on your own dataset. 
For doing so, you need two classes from ``gptchem``: :py:class:`gptchem.tuner.Tuner`
and some ``Formatter``, e.g., :py:class:`gptchem.formatter.ClassificationFormatter`.

For simplicity, we will also use a helper function from the ``data`` module to get a :py:class:`pandas.DataFrame` with the data, but you can also use your own data.

We will also use ``sklearn`` to split the data into training and test set. 
However, depending on the application you might want to use different splitting and 
validation strategies. In any case, make sure to set up some validation procedure to test your model.

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from gptchem.data import get_photoswitch_data
    from gptchem.tuner import Tuner
    from gptchem.formatter import ClassificationFormatter

    # this will download the data upon first use
    data = get_photoswitch_data()

    # encode the data into prompts and completions 
    formatter = ClassificationFormatter(representation_column='SMILES', 
        label_column='E isomer pi-pi* wavelength in nm',
        property_name='E isomer pi-pi* wavelength in nm', 
        num_classes=2,
        qcut=True)

    formatted_data = formatter.format_data(data)
    
    # split the data into training and test set
    train, test = train_test_split(formatted_data, test_size=0.2, random_state=42, stratify=formatted_data['label'])

    # initialize the tuner
    tuner = Tuner()
    tune_summary = tuner(train)

The ``tune_summary`` is a dictionary with the metadata about the tuning procedure as well as ``model_name``, which you will need for querying the model.

Querying the model
--------------------

Once you have a ``model_name`` you can query it with new prompts. 
Convenient helpers for this are provided by the :py:class:`gptchem.querier.Querier` class and extractor classes such as :py:class:`gptchem.extractor.ClassificationExtractor`.

.. code-block:: python 

    from gptchem.querier import Querier
    from gptchem.extractor import ClassificationExtractor

    # initialize the querier
    querier = Querier('ada') # use the model called 'ada'

    # get the completions (assuming the test frame we created above)
    completions = querier(test)

    # extract the predictions
    extractor = ClassificationExtractor()
    predictions = extractor(completions)


Measure the performance
-------------------------

``gptchem`` provides also some helper functions for measuring the performance of the model. :py:func:`gptchem.evaluator.evaluate_classification` is a convenience function for evaluating a classification model.



Logging 
--------------

``gptchem`` uses the `loguru <https://loguru.readthedocs.io/en/stable/index.html>`_  for logging. 
By default, logging from ``gptchem`` is disabled to not interfere with your logs.

However, you can easily customize the logging:

.. code-block:: python

    import sys
    from loguru import logger

    # enable gptchem logging 
    logger.enable("gptchem")
    
    # define the logging level
    LEVEL = "INFO || DEBUG || WARNING || etc."

    # set the handler
    # for logging to stdout
    logger.add(sys.stdout, level=LEVEL) 
    # or for logging to a file
    logger.add("my_log_file.log", level=LEVEL, enqueue=True) 


In many cases, however, you might find it convenient to simply call :py:meth:`~mofdscribe.utils.enable_logging`

.. code-block:: python

    from gptchem.utils import enable_logging

    enable_logging()

which will enable logging with sane defaults (i.e. logging to ``stderr`` for ``INFO`` and ``WARNING`` levels).
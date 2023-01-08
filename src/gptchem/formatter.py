"""
.. admonition:: From the OpenAI Docs:
    :class: note

    To fine-tune a model, you'll need a set of training examples that each consist of a single input ("prompt") and its associated output ("completion"). This is notably different from using our base models, where you might input detailed instructions or multiple examples in a single prompt.

    Each prompt should end with a fixed separator to inform the model when the prompt ends and the completion begins. A simple separator which generally works well is ``\\n\\n###\\n\\n``. The separator should not appear elsewhere in any prompt.
    Each completion should start with a whitespace due to our tokenization, which tokenizes most words with a preceding whitespace.
    Each completion should end with a fixed stop sequence to inform the model when the completion ends. A stop sequence could be ``\\n``, ``###``, or any other token that does not appear in any completion.
    For inference, you should format your prompts in the same way as you did when creating the training dataset, including the same separator. Also specify the same stop sequence to properly truncate the completion.
"""

from typing import List, Optional

import pandas as pd
from fastcore.basics import basic_repr

from .types import StringOrNumber


class BaseFormatter:
    _start_completion = " "
    _stop_sequence = "@@@"
    _end_prompt = "###"
    _prefix = ""
    _suffix = "?"
    _prompt_template = ""
    _completion_template = ""

    def format(self, row: pd.DataFrame) -> dict:
        raise NotImplementedError

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    __repr__ = basic_repr()


class ForwardFormatter(BaseFormatter):
    """Convert a dataframe to a dataframe of prompts and completions for classification or regression.

    The default prompt template is:
        {prefix}What is the {propertyname} of {representation}{suffix}{end_prompt}

    The default completion template is:
        {start_completion}{label}{stop_sequence}

    By default, the following string replacements are made:
        - prefix -> ""
        - suffix -> "?"
        - end_prompt -> "###"
        - start_completion -> " "
        - stop_sequence -> "@@@"
    """

    _PROMPT_TEMPLATE = "{prefix}What is the {propertyname} of {representation}{suffix}{end_prompt}"
    _COMPLETION_TEMPLATE = "{start_completion}{label}{stop_sequence}"

    def format(self) -> dict:
        raise NotImplementedError

    def _format(self, representation: StringOrNumber, label: StringOrNumber) -> dict:
        return {
            "prompt": self._PROMPT_TEMPLATE.format(
                prefix=self._prefix,
                propertyname=self.property_name,
                representation=representation,
                suffix=self._suffix,
                end_prompt=self._end_prompt,
            ),
            "completion": self._COMPLETION_TEMPLATE.format(
                start_completion=self._start_completion,
                label=label,
                stop_sequence=self._stop_sequence,
            ),
            "label": label,
            "representation": representation,
        }

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.format_many(df)


class ClassificationFormatter(ForwardFormatter):
    """Convert a dataframe to a dataframe of prompts and completions for classification.

    The default prompt template is:
        {prefix}What is the {propertyname} of {representation}{suffix}{end_prompt}

    The default completion template is:
        {start_completion}{label}{stop_sequence}

    By default, the following string replacements are made:
        - prefix -> ""
        - suffix -> "?"
        - end_prompt -> "###"
        - start_completion -> " "
        - stop_sequence -> "@@@"

    We map classes to integers, following the advice from
    OpenAI's documentation:

    .. admonition:: From the OpenAI Docs:
        :class: note

        Choose classes that map to a single token.
        At inference time, specify max_tokens=1
        since you only need the first token for classification."
    """

    _PROMPT_TEMPLATE = "{prefix}What is the {propertyname} of {representation}{suffix}{end_prompt}"
    _COMPLETION_TEMPLATE = "{start_completion}{label}{stop_sequence}"

    def __init__(
        self,
        representation_column: str,
        label_column: str,
        property_name: str,
        num_classes: Optional[int] = None,
        qcut: bool = True,
    ) -> None:
        """Initialize a ClassificationFormatter.

        Args:
            representation_column (str): The column name of the representation.
            label_column (str): The column name of the label.
            property_name (str): The name of the property.
            num_classes (int, optional): The number of classes.
            qcut (bool): Whether to use qcut to split the label into classes. Otherwise, cut is used.
        """
        self.representation_column = representation_column
        self.label_column = label_column
        self.num_classes = num_classes
        self.property_name = property_name
        self.qcut = qcut

    __repr__ = basic_repr("representation_column,label_column,property_name,num_classes,qcut")

    @property
    def class_names(self) -> List[int]:
        """Names of the classes."""
        return list(range(self.num_classes))

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format a dataframe of representations and labels into a dataframe of prompts and completions.

        This function will drop rows with missing values in the representation or label columns.

        Args:
            df (pd.DataFrame): A dataframe with a representation column and a label column.

        Returns:
            pd.DataFrame: A dataframe with a prompt column and a completion column.
        """
        df = df.dropna(subset=[self.representation_column, self.label_column])
        representation = df[self.representation_column]
        label = df[self.label_column]

        if self.num_classes is not None:
            if self.qcut:
                label = pd.qcut(label, self.num_classes, labels=self.class_names)
            else:
                label = pd.cut(label, self.num_classes, labels=self.class_names)

        return pd.DataFrame([self._format(r, l) for r, l in zip(representation, label)])


class RegressionFormatter(ForwardFormatter):
    """Convert a dataframe to a dataframe of prompts and completions for regression.

    The default prompt template is:
        {prefix}What is the {propertyname} of {representation}{suffix}{end_prompt}

    The default completion template is:
        {start_completion}{label}{stop_sequence}

    By default, the following string replacements are made:
        - prefix -> ""
        - suffix -> "?"
        - end_prompt -> "###"
        - start_completion -> " "
        - stop_sequence -> "@@@"
    """

    _PROMPT_TEMPLATE = "{prefix}What is the {propertyname} of {representation}{suffix}{end_prompt}"
    _COMPLETION_TEMPLATE = "{start_completion}{label}{stop_sequence}"

    def __init__(
        self,
        representation_column: str,
        label_column: str,
        property_name: str,
        num_digits: int = 2,
    ) -> None:
        """Initialize a ClassificationFormatter.

        Args:
            representation_column (str): The column name of the representation.
            label_column (str): The column name of the label.
            property_name (str): The name of the property.
            num_digits (int): The number of digits to round the label to.
        """
        self.representation_column = representation_column
        self.label_column = label_column
        self.property_name = property_name
        self.num_digits = num_digits

    __repr__ = basic_repr("representation_column,label_column,property_name,num_digits")

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format a dataframe of representations and labels into a dataframe of prompts and completions.

        This function will drop rows with missing values in the representation or label columns.

        Args:
            df (pd.DataFrame): A dataframe with a representation column and a label column.

        Returns:
            pd.DataFrame: A dataframe with a prompt column and a completion column.
        """
        df = df.dropna(subset=[self.representation_column, self.label_column])
        representation = df[self.representation_column]
        label = df[self.label_column]

        label = label.round(self.num_digits)

        return pd.DataFrame([self._format(r, l) for r, l in zip(representation, label)])


class InverseFormatter(BaseFormatter):
    """
    .. admonition:: From the OpenAI Docs:
        :class: note

        Using Lower learning rate and only 1-2 epochs tends to work better for these use cases
    """

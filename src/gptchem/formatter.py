"""
.. admonition:: From the OpenAI Docs:
    :class: note

    To fine-tune a model, you'll need a set of training examples that each consist of a single input ("prompt") and its associated output ("completion"). This is notably different from using our base models, where you might input detailed instructions or multiple examples in a single prompt.

    Each prompt should end with a fixed separator to inform the model when the prompt ends and the completion begins. A simple separator which generally works well is ``\\n\\n###\\n\\n``. The separator should not appear elsewhere in any prompt.
    Each completion should start with a whitespace due to our tokenization, which tokenizes most words with a preceding whitespace.
    Each completion should end with a fixed stop sequence to inform the model when the completion ends. A stop sequence could be ``\\n``, ``###``, or any other token that does not appear in any completion.
    For inference, you should format your prompts in the same way as you did when creating the training dataset, including the same separator. Also specify the same stop sequence to properly truncate the completion.
"""

import random
from typing import Collection, List, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import selfies
import yaml
from fastcore.basics import basic_repr
from loguru import logger
from numpy.typing import ArrayLike
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from selfies import decoder, encoder
from sklearn.preprocessing import LabelEncoder

from .types import StringOrNumber

RDLogger.DisableLog("rdApp.*")


def sanitize_smiles(smi):
    """Return a canonical smile representation of smi

    Parameters:
    smi (string) : smile string to be canonicalized

    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful
    """
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False):
    """Return a mutated selfie string (only one mutation on slefie is performed)

    Mutations are done until a valid molecule is obtained
    Rules of mutation: With a 50% propbabily, either:
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another

    Parameters:
    selfie            (string)  : SELFIE string to be mutated
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"

    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    """
    valid = False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)

    while not valid:
        fail_counter += 1

        alphabet = list(selfies.get_semantic_robust_alphabet())  # 34 SELFIE characters

        choice_ls = [1, 2]  # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]

        # Insert a character in a Random Location
        if random_choice == 1:
            random_index = np.random.randint(len(chars_selfie) + 1)
            random_character = np.random.choice(alphabet, size=1)[0]

            selfie_mutated_chars = (
                chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]
            )

        # Replace a random character
        elif random_choice == 2:
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[random_index + 1 :]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index]
                    + [random_character]
                    + chars_selfie[random_index + 1 :]
                )

        # Delete a random character
        elif random_choice == 3:
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index + 1 :]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index] + chars_selfie[random_index + 1 :]
                )

        else:
            raise Exception("Invalid Operation trying to be performed")

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)

        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon == "":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid = False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write(
                    "Tried to mutate SELFIE: "
                    + str(sf)
                    + " To Obtain: "
                    + str(selfie_mutated)
                    + "\n"
                )
                f.close()

    return (selfie_mutated, smiles_canon)


def get_selfie_chars(selfie):
    """Obtain a list of all selfie characters in string selfie

    Parameters:
    selfie (string) : A selfie string - representing a molecule

    Example:
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    """
    chars_selfie = []  # A list of all SELFIE sybols from string selfie
    while selfie != "":
        chars_selfie.append(selfie[selfie.find("[") : selfie.find("]") + 1])
        selfie = selfie[selfie.find("]") + 1 :]
    return chars_selfie


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
        self.bins = None

    __repr__ = basic_repr("representation_column,label_column,property_name,num_classes,qcut")

    @property
    def class_names(self) -> List[int]:
        """Names of the classes."""
        return list(range(self.num_classes))

    def bin(self, y: ArrayLike):
        """Bin the inputs based on the bins used for the dataset."""
        if self.bins is None:
            raise ValueError("You must fit the formatter first.")

        return pd.cut(y, self.bins, labels=self.class_names, include_lowest=True)

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
                if self.bins is None:
                    _, bins = pd.qcut(list(label.values), self.num_classes, retbins=True)
                    bins = [-np.inf, *bins[1:-1], np.inf]
                    self.bins = bins
                else:
                    bins = self.bins
                label = pd.cut(label, bins=bins, labels=self.class_names, include_lowest=True)

            else:
                if self.bins is None:
                    _, bins = pd.cut(
                        list(label.values) + [np.inf, -np.inf],
                        self.num_classes,
                        retbins=True,
                        include_lowest=True,
                    )
                    self.bins = bins
                else:
                    bins = self.bins

                label = pd.cut(label, bins=bins, labels=self.class_names, include_lowest=True)

        return pd.DataFrame([self._format(r, l) for r, l in zip(representation, label)])


class ClassifictionFormatterWithExamples(ClassificationFormatter):
    _PROMPT_TEMPLATE = (
        """{prefix}What is the {propertyname} of {representation}{suffix}{end_prompt}"""
    )

    _EXAMPLES_TEMPLATE = """

Examples of the prompt/completion structure with dummy data:
##
prompt: {p1}
completion: {c1}
##
prompt: {p2}
completion: {c2}
##
prompt: {p3}
completion: {c3}
    """

    def _format(
        self,
        representation: StringOrNumber,
        label: StringOrNumber,
        possible_labels: Collection[StringOrNumber],
    ) -> dict:
        random_prompts = []
        random_completions = []
        for i in range(3):
            mol = mutate_selfie(selfies.encoder(representation), 500)[1]
            random_completion = random.choice(possible_labels)
            random_prompts.append(
                self._PROMPT_TEMPLATE.format(
                    prefix=self._prefix,
                    propertyname=self.property_name,
                    representation=mol,
                    suffix=self._suffix,
                    end_prompt=self._end_prompt,
                )
            )
            random_completions.append(
                self._COMPLETION_TEMPLATE.format(
                    start_completion=self._start_completion,
                    label=random_completion,
                    stop_sequence=self._stop_sequence,
                ),
            )
        examples = self._EXAMPLES_TEMPLATE.format(
            p1=random_prompts[0],
            p2=random_prompts[1],
            p3=random_prompts[2],
            c1=random_completions[0],
            c2=random_completions[1],
            c3=random_completions[2],
        )

        return {
            "prompt": self._PROMPT_TEMPLATE.format(
                prefix=self._prefix,
                propertyname=self.property_name,
                representation=representation,
                suffix=self._suffix,
                end_prompt=self._end_prompt,
            )
            + examples,
            "completion": self._COMPLETION_TEMPLATE.format(
                start_completion=self._start_completion,
                label=label,
                stop_sequence=self._stop_sequence,
            ),
            "label": label,
            "representation": representation,
        }

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format a dataframe of representations and labels into a dataframe of prompts and completions.

        This function will drop rows with missing values in the representation or label columns.

        Args:
            df (pd.DataFrame): A dataframe with a representation column and a label column.

        Returns:
            pd.DataFrame: A dataframe with a prompt column and a completion column.
        """
        df = df.dropna(subset=[self.representation_column, self.label_column])
        representation = df[self.representation_column].values
        label = df[self.label_column].values

        if self.num_classes is not None:
            if self.qcut:
                if self.bins is None:
                    _, bins = pd.qcut(list(label), self.num_classes, retbins=True)
                    bins = [-np.inf, *bins[1:-1], np.inf]
                    self.bins = bins
                else:
                    bins = self.bins
                label = pd.cut(label, bins=bins, labels=self.class_names, include_lowest=True)

            else:
                if self.bins is None:
                    _, bins = pd.cut(
                        list(label) + [np.inf, -np.inf],
                        self.num_classes,
                        retbins=True,
                        include_lowest=True,
                    )
                    self.bins = bins
                else:
                    bins = self.bins

                label = pd.cut(label, bins=bins, labels=self.class_names, include_lowest=True)

        return pd.DataFrame([self._format(r, l, label) for r, l in zip(representation, label)])


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


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ReactionClassificationFormatter(BaseFormatter):
    _PROMPT_TEMPLATE = (
        "{prefix}What is the {propertyname} of the reaction {representation}{suffix}{end_prompt}"
    )
    _COMPLETION_TEMPLATE = "{start_completion}{label}{stop_sequence}"

    def __init__(
        self,
        reactant_columns: Collection[str],
        reactant_column_names: Collection[str],
        label_column: str,
        property_name: str,
        num_classes: Optional[int] = None,
        qcut: bool = True,
        one_hot: bool = False,
    ) -> None:
        """Initialize a ReactionClassificationFormatter.

        Args:
            reactant_columns (Collection[str]): The column name of the reactants.
            reactant_column_names (Collection[str]): The names of the reactants.
            label_column (str): The column name of the label.
            property_name (str): The name of the property.
            num_classes (int, optional): The number of classes.
            qcut (bool): Whether to use qcut to split the label into classes. Otherwise, cut is used.
            one_hot (bool): Whether to use one hot encoding for the labels.
        """
        self.reactant_columns = reactant_columns
        self.reactant_column_names = reactant_column_names
        self.label_column = label_column
        self.num_classes = num_classes
        self.property_name = property_name
        self.qcut = qcut
        self.bins = None
        self.one_hot = one_hot
        self.le = MultiColumnLabelEncoder(reactant_columns)

    @classmethod
    def from_preset(cls, ds_name, num_classes, one_hot=False, qcut=True):
        benchmarks = {
            "DreherDoyle": {
                "features": ["ligand", "additive", "base", "aryl halide"],
                "feature_names": ["ligand", "additive", "base", "aryl halide"],
                "labels": "yield",
            },
            "DreherDoyleRXN": {
                "features": ["rxn"],
                "labels": "yield",
                "feature_names": ["reaction"],
            },
            "SuzukiMiyaura": {
                "features": [
                    "reactant_1_smiles",
                    "reactant_2_smiles",
                    "catalyst_smiles",
                    "ligand_smiles",
                    "reagent_1_smiles",
                    "solvent_1_smiles",
                ],
                "feature_names": [
                    "reactant 1",
                    "reactant 2",
                    "catalyst",
                    "ligand",
                    "reagent",
                    "solvent",
                ],
                "labels": "yield",
            },
            "SuzukiMiyauraRXN": {
                "features": ["rxn"],
                "labels": "yield",
                "feature_names": ["reaction"],
            },
        }
        if ds_name not in benchmarks:
            raise ValueError(f"Dataset {ds_name} not found.")

        feats = benchmarks[ds_name]["features"]
        label = benchmarks[ds_name]["labels"]
        feat_names = benchmarks[ds_name]["feature_names"]
        return cls(
            reactant_columns=feats,
            label_column=label,
            num_classes=num_classes,
            one_hot=one_hot,
            qcut=qcut,
            reactant_column_names=feat_names,
            property_name="yield",
        )

    @property
    def class_names(self) -> List[int]:
        """Names of the classes."""
        return list(range(self.num_classes))

    def bin(self, y: ArrayLike):
        """Bin the inputs based on the bins used for the dataset."""
        if self.bins is None:
            raise ValueError("You must fit the formatter first.")

        return pd.cut(y, self.bins, labels=self.class_names, include_lowest=True)

    def _representation_string(self, representation):
        return "  ".join([f"{n} {r}" for n, r in zip(self.reactant_column_names, representation)])

    def _format(self, representation: ArrayLike, label: StringOrNumber) -> dict:
        return {
            "prompt": self._PROMPT_TEMPLATE.format(
                prefix=self._prefix,
                propertyname=self.property_name,
                representation=self._representation_string(representation),
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

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format a dataframe of representations and labels into a dataframe of prompts and completions.

        This function will drop rows with missing values in the representation or label columns.

        Args:
            df (pd.DataFrame): A dataframe with a representation column and a label column.

        Returns:
            pd.DataFrame: A dataframe with a prompt column and a completion column.
        """
        df = df.dropna(subset=[self.label_column])
        df = df.fillna(value="None")

        if self.one_hot:
            representation = df[self.reactant_columns]
            representation = self.le.fit_transform(representation).values.tolist()
        else:
            representation = df[self.reactant_columns].values
        representation = list(representation)
        label = df[self.label_column]

        if self.num_classes is not None:
            if self.qcut:
                if self.bins is None:
                    _, bins = pd.qcut(list(label.values), self.num_classes, retbins=True)
                    bins = [-np.inf, *bins[1:-1], np.inf]
                    self.bins = bins
                else:
                    bins = self.bins
                label = pd.cut(label, bins=bins, labels=self.class_names, include_lowest=True)

            else:
                if self.bins is None:
                    _, bins = pd.cut(
                        list(label.values) + [np.inf, -np.inf],
                        self.num_classes,
                        retbins=True,
                        include_lowest=True,
                    )
                    self.bins = bins
                else:
                    bins = self.bins

                label = pd.cut(label, bins=bins, labels=self.class_names, include_lowest=True)

        return pd.DataFrame([self._format(r, l) for r, l in zip(representation, label)])

    __repr__ = basic_repr(
        "reactant_columns, reactant_column_names, label_column, property_name, num_classes, qcut, one_hot"
    )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.format_many(df)


class ReactionRegressionFormatter(BaseFormatter):
    _PROMPT_TEMPLATE = (
        "{prefix}What is the {propertyname} of the reaction {representation}{suffix}{end_prompt}"
    )
    _COMPLETION_TEMPLATE = "{start_completion}{label}{stop_sequence}"

    def __init__(
        self,
        reactant_columns: Collection[str],
        reactant_column_names: Collection[str],
        label_column: str,
        property_name: str,
        num_digit: Optional[int] = None,
        one_hot: bool = False,
    ) -> None:
        """Initialize a ReactionClassificationFormatter.

        Args:
            reactant_columns (Collection[str]): The column name of the reactants.
            reactant_column_names (Collection[str]): The names of the reactants.
            label_column (str): The column name of the label.
            property_name (str): The name of the property.
            num_digit (int, optional): The number of digits to round the label to.
                Defaults to None.
            one_hot (bool): Whether to use one hot encoding for the labels.
        """
        self.reactant_columns = reactant_columns
        self.reactant_column_names = reactant_column_names
        self.label_column = label_column
        self.num_digit = num_digit
        self.property_name = property_name
        self.bins = None
        self.one_hot = one_hot
        self.le = MultiColumnLabelEncoder(reactant_columns)

    @classmethod
    def from_preset(cls, ds_name, num_digit, one_hot=False):
        benchmarks = {
            "DreherDoyle": {
                "features": ["ligand", "additive", "base", "aryl halide"],
                "feature_names": ["ligand", "additive", "base", "aryl halide"],
                "labels": "yield",
            },
            "DreherDoyleRXN": {
                "features": ["rxn"],
                "labels": "yield",
                "feature_names": ["reaction"],
            },
            "SuzukiMiyaura": {
                "features": [
                    "reactant_1_smiles",
                    "reactant_2_smiles",
                    "catalyst_smiles",
                    "ligand_smiles",
                    "reagent_1_smiles",
                    "solvent_1_smiles",
                ],
                "feature_names": [
                    "reactant 1",
                    "reactant 2",
                    "catalyst",
                    "ligand",
                    "reagent",
                    "solvent",
                ],
                "labels": "yield",
            },
            "SuzukiMiyauraRXN": {
                "features": ["rxn"],
                "labels": "yield",
                "feature_names": ["reaction"],
            },
        }
        if ds_name not in benchmarks:
            raise ValueError(f"Dataset {ds_name} not found.")

        feats = benchmarks[ds_name]["features"]
        label = benchmarks[ds_name]["labels"]
        feat_names = benchmarks[ds_name]["feature_names"]
        return cls(
            reactant_columns=feats,
            label_column=label,
            num_digit=num_digit,
            one_hot=one_hot,
            reactant_column_names=feat_names,
            property_name="yield",
        )

    def _representation_string(self, representation):
        return "  ".join([f"{n} {r}" for n, r in zip(self.reactant_column_names, representation)])

    def _format(self, representation: ArrayLike, label: StringOrNumber) -> dict:
        return {
            "prompt": self._PROMPT_TEMPLATE.format(
                prefix=self._prefix,
                propertyname=self.property_name,
                representation=self._representation_string(representation),
                suffix=self._suffix,
                end_prompt=self._end_prompt,
            ),
            "completion": self._COMPLETION_TEMPLATE.format(
                start_completion=self._start_completion,
                label="{:.{prec}f}".format(label, prec=self.num_digit),
                stop_sequence=self._stop_sequence,
            ),
            "label": label,
            "representation": representation,
        }

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format a dataframe of representations and labels into a dataframe of prompts and completions.

        This function will drop rows with missing values in the representation or label columns.

        Args:
            df (pd.DataFrame): A dataframe with a representation column and a label column.

        Returns:
            pd.DataFrame: A dataframe with a prompt column and a completion column.
        """
        df = df.dropna(subset=[self.label_column])
        df = df.fillna(value="None")

        if self.one_hot:
            representation = df[self.reactant_columns]
            representation = self.le.fit_transform(representation).values.tolist()
        else:
            representation = df[self.reactant_columns].values
        representation = list(representation)
        label = df[self.label_column]

        label = label.round(self.num_digit)

        return pd.DataFrame([self._format(r, l) for r, l in zip(representation, label)])

    __repr__ = basic_repr(
        "reactant_columns, reactant_column_names, label_column, property_name, num_classes, qcut, one_hot"
    )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.format_many(df)


class MOFSolventRecommenderFormatter(BaseFormatter):
    _PROMPT_TEMPLATE = (
        "{prefix}In which solution will {linker} and {node}{ion} crystallize{suffix}{end_prompt}"
    )
    _COMPLETION_TEMPLATE = "{start_completion}{label}{stop_sequence}"

    def __init__(
        self,
        linker_columns: List[str],
        node_columns: List[str],
        counter_ion_columns: List[str],
        solvent_columns: List[str],
        solvent_mol_ratio_columns: List[str],
        make_safe: bool = True,
    ):
        self.linker_columns = linker_columns
        self.node_columns = node_columns
        self.solvent_columns = solvent_columns
        self.solvent_mol_ratio_columns = solvent_mol_ratio_columns
        self.counter_ion_columns = counter_ion_columns
        self.make_safe = make_safe

    def _linker_string(self, linker):
        return ", ".join([l for l in linker if not pd.isna(l)])

    def _solvent_string(self, solvent, solvent_mol_ratio):
        return " and ".join(
            [f"{np.round(m,2)} {s}" for s, m in zip(solvent, solvent_mol_ratio) if not np.isnan(m)]
        )

    def _clean(self, string):
        if self.make_safe:
            return quote(string, safe="()=@#?[]").replace("%20", " ")
        return string

    def _format(self, linker, node, ion, solvent, solvent_mol_ratio) -> dict:
        return {
            "prompt": self._clean(
                self._PROMPT_TEMPLATE.format(
                    prefix=self._prefix,
                    linker=self._linker_string(linker),
                    node=str(node[0]).replace("[", "").replace("]", ""),
                    ion=str(ion[0]).replace("[", "").replace("]", ""),
                    suffix=self._suffix,
                    end_prompt=self._end_prompt,
                )
            ),
            "completion": self._clean(
                self._COMPLETION_TEMPLATE.format(
                    start_completion=self._start_completion,
                    label=self._solvent_string(solvent, solvent_mol_ratio),
                    stop_sequence=self._stop_sequence,
                )
            ),
            "label": self._solvent_string(solvent, solvent_mol_ratio),
            "representation": [linker, node, ion, solvent, solvent_mol_ratio],
            "solvents": solvent,
            "solvent_mol_ratios": solvent_mol_ratio,
        }

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format a dataframe of representations and labels into a dataframe of prompts and completions.

        This function will drop rows with missing values in the representation or label columns.

        Args:
            df (pd.DataFrame): A dataframe with a representation column and a label column.

        Returns:
            pd.DataFrame: A dataframe with a prompt column and a completion column.
        """
        # drop entries that have "unknown" in one of the fields
        filtered_rows = []
        df.dropna(subset=[self.linker_columns[0]] + [self.node_columns[0]], inplace=True)
        for _, row in df.iterrows():
            if "unknown" in row[self.counter_ion_columns].values:
                continue
            if any(
                [
                    len(row[linker_col]) > 400
                    for linker_col in self.linker_columns
                    if not pd.isna(row[linker_col])
                ]
            ):
                continue
            filtered_rows.append(row)
        df = pd.DataFrame(filtered_rows)

        linker = df[self.linker_columns].values
        node = df[self.node_columns].values
        ion = df[self.counter_ion_columns].values
        solvent = df[self.solvent_columns].values
        solvent_mol_ratio = df[self.solvent_mol_ratio_columns].values
        return pd.DataFrame(
            [
                self._format(l, n, i, s, smr)
                for l, n, i, s, smr in zip(linker, node, ion, solvent, solvent_mol_ratio)
            ]
        )

    __repr__ = basic_repr(
        "linker_columns, node_columns, counter_ion_columns, solvent_columns, solvent_mol_ratio_columns"
    )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.format_many(df)


class InverseDesignFormatter(BaseFormatter):
    _PROMPT_TEMPLATE = "{prefix}What is a molecule with {property}{suffix}{end_prompt}"
    _COMPLETION_TEMPLATE = "{start_completion}{label}{stop_sequence}"
    _CHECK_NAN = True

    def __init__(
        self,
        representation_column: str,
        property_columns: List[str],
        property_names: List[str],
        num_classes: int = None,
        num_digits: int = 1,
    ):
        self.representation_column = representation_column
        self.property_columns = property_columns
        self.property_names = property_names
        self.num_classes = num_classes
        self.num_digits = num_digits
        self.bins = None

    @property
    def class_names(self) -> List[int]:
        """Names of the classes."""
        return list(range(self.num_classes))

    def bin(self, y: ArrayLike):
        """Bin the inputs based on the bins used for the dataset."""
        if self.bins is None:
            raise ValueError("You must fit the formatter first.")

        return pd.cut(y, self.bins, labels=self.class_names, include_lowest=True)

    def _format_property(self, prop):
        strings = []

        def check_nan(v):
            if self._CHECK_NAN:
                if np.isnan(v):
                    return True
            return False

        for p, v in zip(self.property_names, prop):
            if not check_nan(v):
                if self.num_digits is not None and not self.num_classes:
                    v = np.around(v, self.num_digits)
                    # convert to string with self.num_digits decimal places
                    v = f"{v:.{self.num_digits}f}"

                strings.append(f"{p} {v}")

        return " ,".join(strings)

    def _format(self, representation, prop) -> dict:
        return {
            "prompt": self._PROMPT_TEMPLATE.format(
                prefix=self._prefix,
                property=self._format_property(prop),
                suffix=self._suffix,
                end_prompt=self._end_prompt,
            ),
            "completion": self._COMPLETION_TEMPLATE.format(
                start_completion=self._start_completion,
                label=representation,
                stop_sequence=self._stop_sequence,
            ),
            "label": representation,
            "representation": prop,
        }

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=self.property_columns)
        representation = df[self.representation_column].values
        prop = df[self.property_columns].values

        if self.num_classes is not None:
            if self.bins is None:
                _, bins = pd.qcut(prop.flatten(), self.num_classes, retbins=True)
                bins = [-np.inf, *bins[1:-1], np.inf]
                self.bins = bins
            else:
                bins = self.bins
            prop = pd.cut(
                prop.flatten(), bins=bins, labels=self.class_names, include_lowest=True
            ).astype(int)
            prop = [[p] for p in prop]
        return pd.DataFrame([self._format(r, p) for r, p in zip(representation, prop)])

    __repr__ = basic_repr("representation_column, property_columns, property_names, num_classes")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.format_many(df)


class InverseDesignFormatterWithComposition(InverseDesignFormatter):
    _PROMPT_TEMPLATE = (
        "{prefix}What is a molecule with {property} and {composition}{suffix}{end_prompt}"
    )

    def __init__(
        self,
        representation_column: str,
        property_columns: List[str],
        property_names: List[str],
        num_classes: int = None,
        num_digits: int = 1,
        composition_columns: List[str] = None,
        composition_names: List[str] = None,
    ):
        self.representation_column = representation_column
        self.property_columns = property_columns
        self.property_names = property_names
        self.num_classes = num_classes
        self.num_digits = num_digits
        self.bins = None
        self.composition_columns = composition_columns
        self.composition_names = composition_names

    def _format(self, representation, prop) -> dict:
        return {
            "prompt": self._PROMPT_TEMPLATE.format(
                prefix=self._prefix,
                property=self._format_property(prop),
                suffix=self._suffix,
                end_prompt=self._end_prompt,
            ),
            "completion": self._COMPLETION_TEMPLATE.format(
                start_completion=self._start_completion,
                label=representation,
                stop_sequence=self._stop_sequence,
            ),
            "label": representation,
            "representation": prop,
        }

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=self.property_columns)
        representation = df[self.representation_column].values
        prop = df[self.property_columns].values

        if self.num_classes is not None:
            if self.bins is None:
                _, bins = pd.qcut(prop.flatten(), self.num_classes, retbins=True)
                bins = [-np.inf, *bins[1:-1], np.inf]
                self.bins = bins
            else:
                bins = self.bins
            prop = pd.cut(
                prop.flatten(), bins=bins, labels=self.class_names, include_lowest=True
            ).astype(int)
            prop = [[p] for p in prop]
        return pd.DataFrame([self._format(r, p) for r, p in zip(representation, prop)])


class MOFSynthesisRecommenderFormatter(BaseFormatter):
    _PROMPT_TEMPLATE = "What is the success of a reaction of {ligand} with {salt} in {solvent} {modifier} at {temperature}C for {time}h{end_prompt}"
    _COMPLETION_TEMPLATE = "{start_completion}{label}{stop_sequence}"

    def __init__(
        self,
        ligand_column: Optional[str] = None,
        inorganic_salt_column: Optional[str] = None,
        modifier_column: Optional[str] = None,
        temperature_column: Optional[str] = None,
        time_column: Optional[str] = None,
        solvent_columns: Optional[List[str]] = None,
        solvent_vol_ratio_columns: Optional[List[str]] = None,
        outcome_column: Optional[str] = None,
        score_column: Optional[str] = None,
        doi_column: Optional[str] = None,
        use_score: bool = True,
    ):
        self.ligand_column = ligand_column or "ligand name"
        self.inorganic_salt_column = inorganic_salt_column or "inorganic salt"
        self.modifier_column = modifier_column or "additional"
        self.temperature_column = temperature_column or "T [°C]"
        self.time_column = time_column or "t [h]"
        self.solvent_columns = solvent_columns or ["solvent1", "solvent2", "solvent3"]
        self.solvent_vol_ratio_columns = solvent_vol_ratio_columns or [
            "V/V solvent1 [ ]",
            "V/V solvent2 [ ]",
            "V/V solvent3 [ ]",
        ]
        self.outcome_column = outcome_column or "outcome"
        self.score_column = score_column or "score"
        self.doi_column = doi_column or "reported"
        self.use_score = use_score

    def _solvent_string(self, solvent, solvent_mol_ratio):
        return " and ".join(
            [
                f"{np.round(m,2)} {s}"
                for s, m in zip(solvent, solvent_mol_ratio)
                if not np.isnan(m) and isinstance(s, str) and s != "NA"
            ]
        )

    def _modifier_string(self, modifier):
        if isinstance(modifier, str) and modifier != "NA":
            return f"and {modifier}"
        else:
            return ""

    def _format(
        self, linker, node, solvent, solvent_mol_ratio, modifier, temperature, time, score, outcome
    ) -> dict:
        return {
            "prompt": self._PROMPT_TEMPLATE.format(
                prefix=self._prefix,
                ligand=linker,
                salt=node,
                solvent=self._solvent_string(solvent, solvent_mol_ratio),
                modifier=self._modifier_string(modifier),
                temperature=temperature,
                time=time,
                suffix=self._suffix,
                end_prompt=self._end_prompt,
            ),
            "completion": self._COMPLETION_TEMPLATE.format(
                start_completion=self._start_completion,
                label=score if self.use_score else outcome,
                stop_sequence=self._stop_sequence,
            ),
            "label": score if self.use_score else outcome,
            "representation": [
                linker,
                node,
                solvent,
                solvent_mol_ratio,
                modifier,
                temperature,
                time,
            ],
            "solvents": solvent,
            "solvent_mol_ratios": solvent_mol_ratio,
        }

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format a dataframe of representations and labels into a dataframe of prompts and completions.

        This function will drop rows with missing values in the representation or label columns.

        Args:
            df (pd.DataFrame): A dataframe with a representation column and a label column.

        Returns:
            pd.DataFrame: A dataframe with a prompt column and a completion column.
        """
        # drop entries that have "unknown" in one of the fields
        filtered_rows = []
        df.dropna(subset=[self.ligand_column] + [self.inorganic_salt_column], inplace=True)

        linker = df[self.ligand_column].values
        node = df[self.inorganic_salt_column].values
        solvent = df[self.solvent_columns].values
        solvent_mol_ratio = df[self.solvent_vol_ratio_columns].values
        modifier = df[self.modifier_column].values
        temperature = df[self.temperature_column].values
        time = df[self.time_column].values
        score = df[self.score_column].values
        outcome = df[self.outcome_column].values
        return pd.DataFrame(
            [
                self._format(l, n, s, smr, m, temp, t, sco, out)
                for l, n, s, smr, m, temp, t, sco, out in zip(
                    linker,
                    node,
                    solvent,
                    solvent_mol_ratio,
                    modifier,
                    temperature,
                    time,
                    score,
                    outcome,
                )
            ]
        )

    __repr__ = basic_repr(
        "ligand_column inorganic_salt_column modifier_column temperature_column time_column solvent_columns solvent_vol_ratio_columns outcome_column score_column doi_column use_score"
    )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.format_many(df)


def create_example_string(
    data,
    representation_col: str,
    value_col: str,
    num_examples: Optional[int] = None,
):
    if num_examples is None:
        num_examples = len(data)
    examples = []
    for i, row in data.sample(num_examples).iterrows():
        examples.append(f"Q: {row[representation_col]}\nA: {row[value_col]}\n")
    return "\n".join(examples)


class FewShotFormatter:
    _PREFIX = (
        "I am a highly intelligent question answering bot that answers questions about {property}."
    )
    _PROMPT_TEMPLATE = """{prefix}

{examples}
Q: {representation}"""

    def __init__(
        self,
        training_frame: pd.DataFrame,
        property_name: str,
        representation_column: str,
        label_column: str,
    ):
        self.property_name = property_name
        self.representation_column = representation_column
        self.label_column = label_column
        self.training_frame = training_frame

    __repr__ = basic_repr("representation_column,label_column,property_name")

    def _format(self, row: pd.Series) -> dict:
        """Format a single row of a dataframe into a prompt and completion.

        Args:
            row (pd.Series): A row of a dataframe with a representation and a label.

        Returns:
            dict: A dictionary with a prompt and a completion.
        """

        return {
            "prompt": self._PROMPT_TEMPLATE.format(
                prefix=self._PREFIX.format(property=self.property_name),
                representation=row[self.representation_column],
                examples=create_example_string(
                    self.training_frame,
                    self.representation_column,
                    self.label_column,
                ),
            ),
            "completion": row[self.label_column],
            "label": row[self.label_column],
            "representation": row[self.representation_column],
        }

    def format_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format a dataframe of representations and labels into a dataframe of prompts and completions.

        This function will drop rows with missing values in the representation or label columns.

        Args:
            df (pd.DataFrame): A dataframe with a representation column and a label column.

        Returns:
            pd.DataFrame: A dataframe with a prompt column and a completion column.
        """
        return pd.DataFrame([self._format(row) for _, row in df.iterrows()])

    __call__ = format_many

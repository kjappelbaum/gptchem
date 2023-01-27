import os
import pkgutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Collection, Dict, List, Union
import re
import fcd
import math
import pandas as pd 
import joblib
import numpy as np
import pubchempy as pcp
import pycm
import submitit
from fastcore.all import L
from guacamol.utils.chemistry import (
    calculate_internal_pairwise_similarities,
    calculate_pc_descriptors,
    canonicalize_list,
    continuous_kldiv,
    discrete_kldiv,
    is_valid,
)
from guacamol.utils.data import get_random_subset
from guacamol.utils.sampling_helpers import sample_unique_molecules, sample_valid_molecules
from loguru import logger
from numpy.typing import ArrayLike
from rdkit import Chem, DataStructs
from scipy.optimize import curve_fit, fsolve
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from collections import Counter
from gptchem.models import get_e_pi_pistar_model_data, get_z_pi_pistar_model_data

from .fingerprints.mol_fingerprints import compute_fragprints


POLYMER_FEATURES = [
    "num_[W]",
    "max_[W]",
    "num_[Tr]",
    "max_[Tr]",
    "num_[Ta]",
    "max_[Ta]",
    "num_[R]",
    "max_[R]",
    "[W]",
    "[Tr]",
    "[Ta]",
    "[R]",
    "rel_shannon",
    "length",
]


def evaluate_classification(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Dict[str, Any]:
    """Evaluate a classification task.

    Args:
        y_true (ArrayLike): The true labels.
        y_pred (ArrayLike): The predicted labels.

    Returns:
        Dict[str, Any]: A dictionary of metrics.
    """
    might_have_rounded_floats = False
    assert len(y_true) == len(y_pred), "y_true and y_pred must be the same length."
    y_true = L([int(x) for x in y_true])

    y_pred_new = []
    int_indices = []
    for i, x in enumerate(y_pred):
        try:
            x_int = int(x)
            if x_int != x:
                might_have_rounded_floats = True
                logger.warning("y_pred contains rounded floats.")
            y_pred_new.append(x_int)
            int_indices.append(i)
        except Exception as e:
            y_pred_new.append(None)
    y_pred_new = L(y_pred_new)

    frac_valid = len(int_indices) / len(y_true)
    if len(int_indices) == 0:
        logger.warning("No valid predictions found.")
        y_pred_valid = L([None] * len(y_true))
        y_true_valid = y_true

    else:
        y_true_valid = y_true[int_indices]
        y_pred_valid = y_pred_new[int_indices]

    cm = pycm.ConfusionMatrix(list(y_true_valid), list(y_pred_valid))
    return {
        "accuracy": cm.Overall_ACC,
        "acc_macro": cm.ACC_Macro,
        "racc": cm.Overall_RACC,
        "kappa": cm.Kappa,
        "confusion_matrix": cm,
        "f1_macro": cm.F1_Macro,
        "f1_micro": cm.F1_Micro,
        "frac_valid": frac_valid,
        "all_y_true": y_true,
        "all_y_pred": y_pred,
        "valid_indices": int_indices,
        "might_have_rounded_floats": might_have_rounded_floats,
    }


class KLDivBenchmark:
    """
    Computes the KL divergence between a number of samples and the training set for physchem descriptors.
    Based on the Gucamol implementation
    https://github.com/BenevolentAI/guacamol/blob/8247bbd5e927fbc3d328865d12cf83cb7019e2d6/guacamol/distribution_learning_benchmark.py

    Example:
        >>> from gptchem.evaluator import KLDivBenchmark
        >>> kld = KLDivBenchmark(['CCCC', 'CCCCCC[O]', 'CCCO'], 3)
        >>> kld.score(['CCCCCCCCCCCCC', 'CCC[O]', 'CCCCC'])
        0.5598705863012116
    """

    def __init__(self, training_set: List[str], sample_size: int) -> None:
        """
        Args:
            number_samples: number of samples to generate from the model
            sample_size: molecules from the training set
        """
        self.sample_size = sample_size
        self.training_set_molecules = canonicalize_list(
            get_random_subset(training_set, self.sample_size, seed=42), include_stereocenters=False
        )
        self.pc_descriptor_subset = [
            "BertzCT",
            "MolLogP",
            "MolWt",
            "TPSA",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "NumAliphaticRings",
            "NumAromaticRings",
        ]

    def score(self, molecules: List[str]) -> float:
        """
        Assess a distribution-matching generator model.
        """
        if len(molecules) != self.sample_size:
            logger.warning(
                "The model could not generate enough unique molecules. The score will be penalized."
            )

        # canonicalize_list in order to remove stereo information (also removes duplicates and invalid molecules, but there shouldn't be any)
        unique_molecules = set(canonicalize_list(molecules, include_stereocenters=False))

        # first we calculate the descriptors, which are np.arrays of size n_samples x n_descriptors
        d_sampled = calculate_pc_descriptors(unique_molecules, self.pc_descriptor_subset)
        d_chembl = calculate_pc_descriptors(self.training_set_molecules, self.pc_descriptor_subset)

        kldivs = {}

        # now we calculate the kl divergence for the float valued descriptors ...
        for i in range(4):
            kldiv = continuous_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
            kldivs[self.pc_descriptor_subset[i]] = kldiv

        # ... and for the int valued ones.
        for i in range(4, 9):
            kldiv = discrete_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
            kldivs[self.pc_descriptor_subset[i]] = kldiv

        # pairwise similarity
        chembl_sim = calculate_internal_pairwise_similarities(self.training_set_molecules)
        chembl_sim = chembl_sim.max(axis=1)

        sampled_sim = calculate_internal_pairwise_similarities(unique_molecules)
        sampled_sim = sampled_sim.max(axis=1)

        kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
        kldivs["internal_similarity"] = kldiv_int_int

        # Each KL divergence value is transformed to be in [0, 1].
        # Then their average delivers the final score.
        partial_scores = [np.exp(-score) for score in kldivs.values()]
        score = sum(partial_scores) / len(partial_scores)

        return score


class FrechetBenchmark:
    """Implementation based on the one in Guacamol https://github.com/BenevolentAI/guacamol/blob/master/guacamol/frechet_benchmark.py
    Original implementation

    Example:
        >>> from gptchem.evaluator import FrechetBenchmark
        >>> fb = FrechetBenchmark(['CCC', 'C[O]'], sample_size=2)
        >>> fb.score(['CCC', 'CCCCC'])
        (8.597685719787249, 0.17914904894421219)
    """

    def __init__(
        self,
        training_set: List[str],
        chemnet_model_filename="ChemNet_v0.13_pretrained.h5",
        sample_size=10000,
    ) -> None:
        """
        Args:
            training_set: molecules from the training set
            chemnet_model_filename: name of the file for trained ChemNet model.
                Must be present in the 'fcd' package, since it will be loaded directly from there.
            sample_size: how many molecules to generate the distribution statistics from (both reference data and model)
        """
        self.chemnet_model_filename = chemnet_model_filename
        self.sample_size = sample_size

        self.reference_molecules = get_random_subset(training_set, self.sample_size, seed=42)

    def _load_chemnet(self):
        """
        Load the ChemNet model from the file specified in the init function.
        This file lives inside a package but to use it, it must always be an actual file.
        The safest way to proceed is therefore:
        1. read the file with pkgutil
        2. save it to a temporary file
        3. load the model from the temporary file
        """
        model_bytes = pkgutil.get_data("fcd", self.chemnet_model_filename)
        assert model_bytes is not None

        tmpdir = tempfile.gettempdir()
        model_path = os.path.join(tmpdir, self.chemnet_model_filename)

        with open(model_path, "wb") as f:
            f.write(model_bytes)

        logger.info(f"Saved ChemNet model to '{model_path}'")

        return fcd.load_ref_model(model_path)

    def score(self, generated_molecules: List[str]):
        chemnet = self._load_chemnet()

        mu_ref, cov_ref = self._calculate_distribution_statistics(chemnet, self.reference_molecules)
        mu, cov = self._calculate_distribution_statistics(chemnet, generated_molecules)

        frechet_distance = fcd.calculate_frechet_distance(
            mu1=mu_ref, mu2=mu, sigma1=cov_ref, sigma2=cov
        )
        score = np.exp(-0.2 * frechet_distance)

        return frechet_distance, score

    def _calculate_distribution_statistics(self, model, molecules: List[str]):
        sample_std = fcd.canonical_smiles(molecules)
        gen_mol_act = fcd.get_predictions(model, sample_std)

        mu = np.mean(gen_mol_act, axis=0)
        cov = np.cov(gen_mol_act.T)
        return mu, cov


def get_similarity_to_train_mols(smiles: str, train_smiles: List[str]) -> List[float]:
    train_mols = [Chem.MolFromSmiles(x) for x in train_smiles]
    mol = Chem.MolFromSmiles(smiles)

    train_fps = [Chem.RDKFingerprint(x) for x in train_mols]
    fp = Chem.RDKFingerprint(mol)

    s = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return s


def predict_photoswitch(
    smiles: Collection[str],
):
    """Predicting for a single SMILES string. Not really efficient due to the I/O overhead in loading the model."""
    if not isinstance(smiles, (list, tuple)):
        smiles = [smiles]
    e_pi_pi_star_model = joblib.load(get_e_pi_pistar_model_data())
    z_pi_pi_star_model = joblib.load(get_z_pi_pistar_model_data())
    fragprints = compute_fragprints(smiles)
    return e_pi_pi_star_model.predict(fragprints), z_pi_pi_star_model.predict(fragprints)


def is_valid_smiles(smiles: str) -> bool:
    """We say a SMILES is valid if RDKit can parse it."""
    return is_valid(smiles)


def is_in_pubchem(smiles):
    """Check if a SMILES is in PubChem."""
    try:
        return pcp.get_compounds(smiles, smiles=smiles, namespace="SMILES")
    except Exception:
        return None

def evaluate_generated_smiles(
    smiles: Collection[str], train_smiles: Collection[str]
) -> Dict[str, float]:
    valid_smiles = []
    valid_indices = []
    for i, s in enumerate(smiles):
        if is_valid_smiles(s):
            valid_smiles.append(s)
            valid_indices.append(i)

    assert len(valid_smiles) == len(valid_indices) <= len(smiles)

    frac_valid = len(valid_smiles) / len(smiles)

    unique_smiles = list(set(valid_smiles))

    try:
        kld_bench = KLDivBenchmark(train_smiles, min(len(train_smiles), len(unique_smiles)))
        kld = kld_bench.score(unique_smiles)
    except Exception as e:
        print(e)
        kld = np.nan

    try:
        fb  = FrechetBenchmark(
            train_smiles, sample_size=min(len(train_smiles), len(unique_smiles))
        )
        frechet_d, frechet_score = fb.score(unique_smiles)
    except Exception:
        frechet_d, frechet_score = np.nan, np.nan

    try:
        frac_unique = len(unique_smiles) / len(valid_smiles)
    except ZeroDivisionError:
        frac_unique = 0.0


    frac_smiles_in_train = len([x for x in valid_smiles if x in train_smiles]) / len(valid_smiles)

    in_pubchem = []#[i, x for x in enumerate(valid_smiles) if is_in_pubchem(x)]
    check_ok = 0
    for i, x in enumerate(valid_smiles):
        res=  is_in_pubchem(x)
        if res is not None:
            check_ok += 1
        if res:
            in_pubchem.append(i)

    try:
        frac_smiles_in_pubchem = len(in_pubchem) / check_ok
    except ZeroDivisionError:
        frac_smiles_in_pubchem = 0.0
    
    res = {
        "frac_valid": frac_valid,
        "frac_unique": frac_unique,
        "frac_smiles_in_train": frac_smiles_in_train,
        "frac_smiles_in_pubchem": frac_smiles_in_pubchem,
        "in_pubchem": in_pubchem,
        "kld": kld,
        "frechet_d": frechet_d,
        "frechet_score": frechet_score,
        "valid_smiles": valid_smiles,
        "valid_indices": valid_indices,
    }

    return res


def get_regression_metrics(
    y_true,  # actual values (ArrayLike)
    y_pred,  # predicted values (ArrayLike)
) -> dict:

    try:
        return {
            "r2": r2_score(y_true, y_pred),
            "max_error": max_error(y_true, y_pred),
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
            "mean_squared_error": mean_squared_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
        }
    except Exception:
        return {
            "r2": np.nan,
            "max_error": np.nan,
            "mean_absolute_error": np.nan,
            "mean_squared_error": np.nan,
            "rmse": np.nan,
        }


def evaluate_photoswitch_smiles_pred(
    smiles, expected_z_pi_pi_star, expected_e_pi_pi_star
) -> Dict[str, Dict[str, float]]:
    """Evaluate the predicted photoswitch properties."""
    if smiles:
        pred_e_pi_pi_star, pred_z_pi_pi_star = predict_photoswitch(smiles)
        pred_e_pi_pi_star = np.array(pred_e_pi_pi_star).flatten()
        pred_z_pi_pi_star = np.array(pred_z_pi_pi_star).flatten()
        e_pi_pi_star_metrics = get_regression_metrics(expected_e_pi_pi_star, pred_e_pi_pi_star)
        z_pi_pi_star_metrics = get_regression_metrics(expected_z_pi_pi_star, pred_z_pi_pi_star)
    else: 
        e_pi_pi_star_metrics = {
            "r2": np.nan,
            "max_error": np.nan,
            "mean_absolute_error": np.nan,
            "mean_squared_error": np.nan,
            "rmse": np.nan,
        }
        z_pi_pi_star_metrics = {
            "r2": np.nan,
            "max_error": np.nan,
            "mean_absolute_error": np.nan,
            "mean_squared_error": np.nan,
            "rmse": np.nan,
        }
    return {
        "e_pi_pi_star_metrics": e_pi_pi_star_metrics,
        "z_pi_pi_star_metrics": z_pi_pi_star_metrics,
    }


def get_homo_lumo_gap(file):
    with open(file) as f:
        lines = f.readlines()
    for line in lines[::-1]:
        if "HOMO-LUMO" in line:
            return float(line.split()[-3])
    return None


def get_xtb_homo_lumo_gap(smiles: str) -> float:
    """Run the following to commands in sequence

    givemeconformer "{smiles}"
    xtb conformers.sdf --opt tight > xtb.out
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = f"givemeconformer '{smiles}'"
        subprocess.run(cmd, shell=True, check=True, cwd=tmpdir)
        cmd = "xtb conformers.sdf --opt tight > xtb.out"
        subprocess.run(cmd, shell=True, check=True, cwd=tmpdir)
        gap = get_homo_lumo_gap(Path(tmpdir) / "xtb.out")
    return gap


def get_homo_lump_gaps(
    smiles: Collection[str],
    max_parallel: int = 32,
    timeout: int = 60 * 1,
    partition: str = "LocalQ",
    debug: bool = False,
) -> List[float]:
    """Use submitit to get the HOMO-LUMO gaps of the SMILES.

    This script will assume that you have slurm
    as well as ``givemeconformer`` and ``xtb`` installed.

    The function will run until all jobs are done or all
    the timeouts are reached.

    Args:
        smiles (Collection[str]): A collection of SMILES
        max_parallel (int, optional): The maximum number of jobs to run in parallel.
            Defaults to 32.
        timeout (int, optional): The timeout in minutes. Defaults to 60 * 1.
        partition (str, optional): The partition to run on. Defaults to "LocalQ".
        debug (bool, optional): If True, will run locally. Defaults to False.

    Returns:
        List of floats
    """
    if debug:
        executor = submitit.LocalExecutor(folder="submitit_jobs")
    else:
        executor = submitit.AutoExecutor(folder="submitit_jobs")
    executor.update_parameters(
        timeout_min=timeout,
        slurm_partition=partition,
        slurm_array_parallelism=max_parallel,
        cpus_per_task=1,
        gpus_per_node=0,
        slurm_signal_delay_s=120,
    )
    jobs = executor.map_array(get_xtb_homo_lumo_gap, smiles)
    results = []
    for job in jobs:
        try:
            res = job.result()
        except Exception as e:
            print(e)
            res = None
        results.append(res)
    return results


def lc(x, a, b, c):
    return -a * np.exp(-b * x) + c


def fit_learning_curve(num_training_points: Collection[float], performance: Collection[float]):
    """Fit a learning curve to the performance.

    We force the fit to go through the origin.

    Parameters:
        num_training_points (Collection[float]): list of number of training points
        performance (Collection[float]): list of performance values

    Returns:
        popt (np.ndarray): the parameters of the fit
        pcov (np.ndarray): the covariance matrix of the fit
    """
    num_training_points = [0] + list(num_training_points)
    performance = [0] + list(performance)
    popt, pcov = curve_fit(lc, num_training_points, performance, full_output=False)
    return popt, pcov


def find_learning_curve_intersection(performance, popt):
    """Find the intersection of the learning curve with the performance."""

    def func2(x):
        return lc(x, *popt) - performance

    return fsolve(func2, 0)[0]


def evaluate_homo_lumo_gap(
    smiles: Collection[str],
    expected_gaps: Collection[float],
    get_homo_lumo_gaps_kwargs: Dict = {},
) -> Dict[str, float]:
    computed_gaps = get_homo_lump_gaps(smiles, **get_homo_lumo_gaps_kwargs)

    computed_not_none = []
    expected_not_none = []
    for gap, expected in zip(computed_gaps, expected_gaps):
        if gap is not None:
            computed_not_none.append(gap)
            expected_not_none.append(expected)

    metrics = get_regression_metrics(expected_not_none, computed_not_none)
    metrics['computed_gaps'] = computed_gaps
    return metrics



class PolymerKLDivBenchmark:
    """
    Computes the KL divergence between a number of samples and the training set for physchem descriptors.
    Based on the Gucamol implementation
    https://github.com/BenevolentAI/guacamol/blob/8247bbd5e927fbc3d328865d12cf83cb7019e2d6/guacamol/distribution_learning_benchmark.py
    """

    def __init__(self, training_set: pd.DataFrame, number_samples: int) -> None:
        """
        Args:
            number_samples: number of samples to generate from the model
            training_set: molecules from the training set
        """
        self.number_samples = number_samples
        self.training_set = training_set.sample(n=self.number_samples, random_state=42)

    def score(self, test_set: pd.DataFrame) -> float:
        """
        Assess a distribution-matching generator model.
        """
        test_set = test_set.sample(n=self.number_samples, random_state=42)


        kldivs = {}

        # now we calculate the kl divergence for the float valued descriptors ...
        for i, feat in enumerate(POLYMER_FEATURES):
            kldiv = continuous_kldiv(
                X_baseline=self.training_set[feat], X_sampled=test_set[feat]
            )
            kldivs[feat] = kldiv

        # Each KL divergence value is transformed to be in [0, 1].
        # Then their average delivers the final score.
        partial_scores = [np.exp(-score) for score in kldivs.values()]
        score = sum(partial_scores) / len(partial_scores)

        return score



def get_polymer_completion_composition(string):
    parts = string.split("-")
    counts = Counter(parts)
    return dict(counts)

def convert2smiles(string):
    new_encoding = {"A": "[Ta]", "B": "[Tr]", "W": "[W]", "R": "[R]"}

    for k, v in new_encoding.items():
        string = string.replace(k, v)

    string = string.replace("-", "")

    return string



class LinearPolymerSmilesFeaturizer:
    """Compute features for linear polymers"""

    def __init__(self, smiles: str, normalized_cluster_stats: bool = True):
        self.smiles = smiles
        assert "(" not in smiles, "This featurizer does not work for branched polymers"
        self.characters = ["[W]", "[Tr]", "[Ta]", "[R]"]
        self.replacement_dict = dict(
            list(zip(self.characters, ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]))
        )
        self.normalized_cluster_stats = normalized_cluster_stats
        self.surface_interactions = {"[W]": 30, "[Ta]": 20, "[Tr]": 30, "[R]": 20}
        self.solvent_interactions = {"[W]": 30, "[Ta]": 25, "[Tr]": 35, "[R]": 30}
        self._character_count = None
        self._balance = None
        self._relative_shannon = None
        self._cluster_stats = None
        self._head_tail_feat = None
        self.features = None

    @staticmethod
    def get_head_tail_features(string: str, characters: list) -> dict:
        """0/1/2 encoded feature indicating if the building block is at start/end of the polymer chain"""
        is_head_tail = [0] * len(characters)

        for i, char in enumerate(characters):
            if string.startswith(char):
                is_head_tail[i] += 1
            if string.endswith(char):
                is_head_tail[i] += 1

        new_keys = ["head_tail_" + char for char in characters]
        return dict(list(zip(new_keys, is_head_tail)))

    @staticmethod
    def get_cluster_stats(
        s: str, replacement_dict: dict, normalized: bool = True
    ) -> dict:  # pylint:disable=invalid-name
        """Statistics describing clusters such as [Tr][Tr][Tr]"""
        clusters = LinearPolymerSmilesFeaturizer.find_clusters(s, replacement_dict)
        cluster_stats = {}
        cluster_stats["total_clusters"] = 0
        for key, value in clusters.items():
            if value:
                cluster_stats["num" + "_" + key] = len(value)
                cluster_stats["total_clusters"] += len(value)
                cluster_stats["max" + "_" + key] = max(value)
                cluster_stats["min" + "_" + key] = min(value)
                cluster_stats["mean" + "_" + key] = np.mean(value)
            else:
                cluster_stats["num" + "_" + key] = 0
                cluster_stats["max" + "_" + key] = 0
                cluster_stats["min" + "_" + key] = 0
                cluster_stats["mean" + "_" + key] = 0

        if normalized:
            for key, value in cluster_stats.items():
                if "num" in key:
                    try:
                        cluster_stats[key] = value / cluster_stats["total_clusters"]
                    except ZeroDivisionError:
                        cluster_stats[key] = 0

        return cluster_stats

    @staticmethod
    def find_clusters(s: str, replacement_dict: dict) -> dict:  # pylint:disable=invalid-name
        """Use regex to find clusters"""
        clusters = re.findall(
            r"((\w)\2{1,})", LinearPolymerSmilesFeaturizer._multiple_replace(s, replacement_dict)
        )
        cluster_dict = dict(
            list(zip(replacement_dict.keys(), [[] for i in replacement_dict.keys()]))
        )
        inv_replacement_dict = {v: k for k, v in replacement_dict.items()}
        for cluster, character in clusters:
            cluster_dict[inv_replacement_dict[character]].append(len(cluster))

        return cluster_dict

    @staticmethod
    def _multiple_replace(s: str, replacement_dict: dict) -> str:  # pylint:disable=invalid-name
        for word in replacement_dict:
            s = s.replace(word, replacement_dict[word])
        return s

    @staticmethod
    def get_counts(smiles: str, characters: list) -> dict:
        """Count characters in SMILES string"""
        counts = [smiles.count(char) for char in characters]
        return dict(list(zip(characters, counts)))

    @staticmethod
    def get_relative_shannon(character_count: dict) -> float:
        """Shannon entropy of string relative to maximum entropy of a string of the same length"""
        counts = [c for c in character_count.values() if c > 0]
        length = sum(counts)
        probs = [count / length for count in counts]
        ideal_entropy = LinearPolymerSmilesFeaturizer._entropy_max(length)
        entropy = -sum([p * math.log(p) / math.log(2.0) for p in probs])

        return entropy / ideal_entropy

    @staticmethod
    def _entropy_max(length: int) -> float:
        "Calculates the max Shannon entropy of a string with given length"

        prob = 1.0 / length

        return -1.0 * length * prob * math.log(prob) / math.log(2.0)

    @staticmethod
    def get_balance(character_count: dict) -> dict:
        """Frequencies of characters"""
        counts = list(character_count.values())
        length = sum(counts)
        frequencies = [c / length for c in counts]
        return dict(list(zip(character_count.keys(), frequencies)))

    def _featurize(self):
        """Run all available featurization methods"""
        self._character_count = LinearPolymerSmilesFeaturizer.get_counts(
            self.smiles, self.characters
        )
        self._balance = LinearPolymerSmilesFeaturizer.get_balance(self._character_count)
        self._relative_shannon = LinearPolymerSmilesFeaturizer.get_relative_shannon(
            self._character_count
        )
        self._cluster_stats = LinearPolymerSmilesFeaturizer.get_cluster_stats(
            self.smiles, self.replacement_dict, self.normalized_cluster_stats
        )
        self._head_tail_feat = LinearPolymerSmilesFeaturizer.get_head_tail_features(
            self.smiles, self.characters
        )

        self.features = self._head_tail_feat
        self.features.update(self._cluster_stats)
        self.features.update(self._balance)
        self.features["rel_shannon"] = self._relative_shannon
        self.features["length"] = sum(self._character_count.values())
        solvent_interactions = sum(
            [
                [self.solvent_interactions[char]] * count
                for char, count in self._character_count.items()
            ],
            [],
        )
        self.features["total_solvent"] = sum(solvent_interactions)
        self.features["std_solvent"] = np.std(solvent_interactions)
        surface_interactions = sum(
            [
                [self.surface_interactions[char]] * count
                for char, count in self._character_count.items()
            ],
            [],
        )
        self.features["total_surface"] = sum(surface_interactions)
        self.features["std_surface"] = np.std(surface_interactions)

    def featurize(self) -> dict:
        """Run featurization"""
        self._featurize()
        return self.features

def featurize_many_polymers(smiless: list) -> pd.DataFrame:
    """Utility function that runs featurizaton on a
    list of linear polymer smiles and returns a dataframe"""
    features = []
    for smiles in smiless:
        pmsf = LinearPolymerSmilesFeaturizer(smiles)
        features.append(pmsf.featurize())
    return pd.DataFrame(features)

def polymer_string2performance(string, model_dir = '../models'):
    DELTA_G_MODEL = joblib.load(os.path.join(model_dir, 'delta_g_model.joblib'))

    predicted_monomer_sequence = string.split("@")[0].strip()
    monomer_sq = re.findall("[(R|W|A|B)\-(R|W|A|B)]+", predicted_monomer_sequence)[0]
    composition = get_polymer_completion_composition(monomer_sq)
    smiles = convert2smiles(predicted_monomer_sequence)

    features = pd.DataFrame(featurize_many_polymers([smiles]))
    prediction = DELTA_G_MODEL.predict(features[POLYMER_FEATURES])
    return {
        "monomer_squence": monomer_sq,
        "composition": composition,
        "smiles": smiles,
        "prediction": prediction,
        'features': features
    }
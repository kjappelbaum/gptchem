import os
import pkgutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Collection, Dict, List, Union

import fcd
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

from gptchem.models import get_e_pi_pistar_model_data, get_z_pi_pistar_model_data

from .fingerprints.mol_fingerprints import compute_fragprints


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
    try:
        kld_bench = KLDivBenchmark(train_smiles, min(len(train_smiles), len(valid_smiles)))
        kld = kld_bench.score(valid_smiles)
    except Exception:
        kld = np.nan

    try:
        frechet_d, frechet_score = FrechetBenchmark(
            train_smiles, sample_size=min(len(train_smiles), len(valid_smiles))
        )
    except Exception:
        frechet_d, frechet_score = np.nan, np.nan

    unique_smiles = list(set(valid_smiles))
    frac_unique = len(unique_smiles) / len(valid_smiles)

    frac_smiles_in_train = len([x for x in valid_smiles if x in train_smiles]) / len(valid_smiles)

    in_pubchem = []#[i, x for x in enumerate(valid_smiles) if is_in_pubchem(x)]
    check_ok = 0
    for i, x in enumerate(valid_smiles):
        res=  is_in_pubchem(x)
        if res is not None:
            check_ok += 1
        if res:
            in_pubchem.append(i)
    
    frac_smiles_in_pubchem = len(in_pubchem) / check_ok

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
        cmd = f"givemeconformer {smiles}"
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

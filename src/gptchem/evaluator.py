import os
import pkgutil
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycm
from fastcore.all import L
from loguru import logger
from numpy.typing import ArrayLike
from rdkit import Chem, DataStructs
from rdkit.Contrib.SA_Score.sascorer import calculateScore as calculate_sascore
from scipy.optimize import curve_fit, fsolve
from scipy.stats import entropy, gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from gptchem.fingerprints.polymer import featurize_many_polymers
from gptchem.models import get_e_pi_pistar_model_data, get_polymer_model, get_z_pi_pistar_model_data

from .fingerprints.mol_fingerprints import compute_fragprints


def continuous_kldiv(X_baseline: np.ndarray, X_sampled: np.ndarray, pca: bool = False) -> float:
    """Calculate the continuous Kullback-Leibler divergence between two distributions."""
    if pca:
        pca = PCA(n_components=1)
        X_baseline = pca.fit_transform(X_baseline.reshape(-1, 1)).flatten()
        X_sampled = pca.transform(X_sampled.reshape(-1, 1)).flatten()
    kde_P = gaussian_kde(X_baseline)
    kde_Q = gaussian_kde(X_sampled)
    x_eval = np.linspace(
        np.hstack([X_baseline, X_sampled]).min(), np.hstack([X_baseline, X_sampled]).max(), num=1000
    )
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10

    return entropy(P, Q)


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
        from guacamol.utils.chemistry import canonicalize_list
        from guacamol.utils.data import get_random_subset

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
        from guacamol.utils.chemistry import (
            calculate_internal_pairwise_similarities,
            calculate_pc_descriptors,
            canonicalize_list,
            discrete_kldiv,
        )

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
        from guacamol.utils.data import get_random_subset

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
        import fcd

        model_bytes = pkgutil.get_data("fcd", self.chemnet_model_filename)
        assert model_bytes is not None

        tmpdir = tempfile.gettempdir()
        model_path = os.path.join(tmpdir, self.chemnet_model_filename)

        with open(model_path, "wb") as f:
            f.write(model_bytes)

        logger.info(f"Saved ChemNet model to '{model_path}'")

        return fcd.load_ref_model(model_path)

    def score(self, generated_molecules: List[str]):
        import fcd

        chemnet = self._load_chemnet()

        mu_ref, cov_ref = self._calculate_distribution_statistics(chemnet, self.reference_molecules)
        mu, cov = self._calculate_distribution_statistics(chemnet, generated_molecules)

        frechet_distance = fcd.calculate_frechet_distance(
            mu1=mu_ref, mu2=mu, sigma1=cov_ref, sigma2=cov
        )
        score = np.exp(-0.2 * frechet_distance)

        return frechet_distance, score

    def _calculate_distribution_statistics(self, model, molecules: List[str]):
        import fcd

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


def get_sa_scores(smiles):
    sa_scores = []
    for smiles in smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            sa_scores.append(calculate_sascore(mol))
        except:
            pass
    return sa_scores


def is_valid_smiles(smiles: str) -> bool:
    """We say a SMILES is valid if RDKit can parse it."""
    from guacamol.utils.chemistry import is_valid

    return is_valid(smiles)


@lru_cache(maxsize=None)
def is_in_pubchem(smiles):
    """Check if a SMILES is in PubChem."""
    import pubchempy as pcp

    try:
        res = pcp.get_compounds(smiles, smiles=smiles, namespace="SMILES")
        return (len(res) > 0) & (res[0].cid is not None)
    except Exception as e:
        print(e)
        return False


def evaluate_generated_smiles(
    smiles: Collection[str], train_smiles: Collection[str]
) -> Dict[str, float]:
    valid_smiles = []
    valid_indices = []
    novel_indices = []
    novel_smiles = []
    for i, s in enumerate(smiles):
        s = s.split()[0].strip()
        if is_valid_smiles(s):
            valid_smiles.append(s)
            valid_indices.append(i)
            if s not in train_smiles:
                novel_indices.append(i)
                novel_smiles.append(s)

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
        fb = FrechetBenchmark(train_smiles, sample_size=min(len(train_smiles), len(unique_smiles)))
        frechet_d, frechet_score = fb.score(unique_smiles)
    except Exception:
        frechet_d, frechet_score = np.nan, np.nan

    try:
        frac_unique = len(unique_smiles) / len(valid_smiles)
    except ZeroDivisionError:
        frac_unique = 0.0

    frac_smiles_in_train = len([x for x in valid_smiles if x in train_smiles]) / len(valid_smiles)

    in_pubchem = []  # [i, x for x in enumerate(valid_smiles) if is_in_pubchem(x)]
    check_ok = 0
    for i, x in enumerate(valid_smiles):
        res = is_in_pubchem(x)
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
        "novel_indices": novel_indices,
        "novel_smiles": novel_smiles,
    }

    return res


def get_regression_metrics(
    y_true: np.typing.ArrayLike,
    y_pred: np.typing.ArrayLike,
) -> dict:
    """Compute regression metrics."""
    try:
        return {
            "r2": r2_score(y_true, y_pred),
            "max_error": max_error(y_true, y_pred),
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
            "mean_squared_error": mean_squared_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "mean_absolute_percentage_error": mean_absolute_percentage_error(y_true, y_pred),
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
        "expected_e_pi_pi_star": expected_e_pi_pi_star,
        "expected_z_pi_pi_star": expected_z_pi_pi_star,
    }


def get_homo_lumo_gap(file: Union[Path, str]):
    """Parse HOMO-LUMO gap from xtb output file."""
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
    from diskcache import Cache

    CACHE_DIR = os.getenv("CACHEDIR", "gptchemcache")

    # 2 ** 30 = 1 GB
    gap_cache = Cache(CACHE_DIR, size_limit=2**30, disk_min_file_size=0)

    try:
        gap = gap_cache.get(smiles)
        if gap is not None:
            return gap
    except KeyError:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = f"givemeconformer '{smiles}'"
        subprocess.run(cmd, shell=True, check=True, cwd=tmpdir)
        cmd = "xtb conformers.sdf --opt tight > xtb.out"
        subprocess.run(cmd, shell=True, check=True, cwd=tmpdir)
        gap = get_homo_lumo_gap(Path(tmpdir) / "xtb.out")
        gap_cache[smiles] = gap
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
    import submitit

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
    """Learning curve function."""
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
    metrics["computed_gaps"] = computed_gaps
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

    def score(self, test_polymers: Collection[str]) -> float:
        """
        Assess a distribution-matching generator model.
        """
        test_set = featurize_many_polymers([polymer_convert2smiles(p) for p in test_polymers])
        test_set = test_set.sample(n=self.number_samples, random_state=42)

        kldivs = {}

        # now we calculate the kl divergence for the float valued descriptors ...
        for i, feat in enumerate(POLYMER_FEATURES):
            kldiv = continuous_kldiv(
                X_baseline=self.training_set[feat].values, X_sampled=test_set[feat].values, pca=True
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


def polymer_convert2smiles(string: str) -> str:
    """Converts a polymer string to a polymer SMILES string."""
    new_encoding = {"A": "[Ta]", "B": "[Tr]", "W": "[W]", "R": "[R]"}

    for k, v in new_encoding.items():
        string = string.replace(k, v)

    string = string.replace("-", "")

    return string


def polymer_string2performance(string: str) -> dict:
    """Converts a polymer string to a performance dictionary.

    Args:
        string (str): polymer string

    Returns:
        dict: performance dictionary

    Example:
        >>> res = polymer_string2performance("W-A-B-W-W-A-A-A-R-W-B-B-R-R-B-R")
        assert 'prediction' in res
        assert 'composition' in res
    """
    DELTA_G_MODEL = joblib.load(get_polymer_model())

    predicted_monomer_sequence = string.split("@")[0].strip()
    monomer_sq = re.findall("[(R|W|A|B)\-(R|W|A|B)]+", predicted_monomer_sequence)[0]
    composition = get_polymer_completion_composition(monomer_sq)
    smiles = polymer_convert2smiles(predicted_monomer_sequence)

    features = pd.DataFrame(featurize_many_polymers([smiles]))
    prediction = DELTA_G_MODEL.predict(features[POLYMER_FEATURES])
    return {
        "monomer_squence": monomer_sq,
        "composition": composition,
        "smiles": smiles,
        "prediction": prediction,
        "features": features,
    }


def composition_mismatch(composition: dict, found: dict) -> dict:
    """Calculate the distance between the composition and the found composition.
    Used for the polymer completion task.

    Args:
        composition (dict): The expected composition
        found (dict): The found composition

    Returns:
        dict: A dictionary with the distances, the min, max, mean and the expected length

    Example:
        >>> composition_mismatch(
            {"A": 4, "B": 4, "R": 12, "W": 12},
            {'W': 12, 'R': 12, 'A': 4, 'B': 5}
        )
        {'distances': [0, 0, 0, 1],
        'min': 0,
        'max': 1,
        'mean': 0.25,
        'expected_len': 32,
        'found_len': 33}
    """
    distances = []

    # We also might have the case the there are keys that the input did not contain
    all_keys = set(composition.keys()) & set(found.keys())

    expected_len = []
    found_len = []

    for key in all_keys:
        try:
            expected = composition[key]
        except KeyError:
            expected = 0
        expected_len.append(expected)
        try:
            f = found[key]
        except KeyError:
            f = 0
        found_len.append(f)

        distances.append(np.abs(expected - f))

    expected_len = sum(expected_len)
    found_len = sum(found_len)
    return {
        "distances": distances,
        "min": np.min(distances),
        "max": np.max(distances),
        "mean": np.mean(distances),
        "expected_len": expected_len,
        "found_len": found_len,
    }


def string_distances(training_set: Collection[str], query_string: str) -> dict:
    """Calculate the distances between the query string and the training set.

    Args:
        training_set (Collection[str]): The training set
        query_string (str): The query string

    Returns:
        dict: A dictionary with the distances, the min, max, mean and the expected length

    Example:
        >>> training_set = ["AAA", "BBB", "CCC"]
        >>> query_string = "BBB"
        >>> result = string_distances(training_set, query_string)
        assert result["NormalizedLevenshtein_min"] == 0.0
        assert result["NormalizedLevenshtein_max"] == 1.0
    """
    from strsimpy.levenshtein import Levenshtein
    from strsimpy.longest_common_subsequence import LongestCommonSubsequence
    from strsimpy.normalized_levenshtein import NormalizedLevenshtein

    distances = defaultdict(list)

    metrics = [
        ("Levenshtein", Levenshtein()),
        ("NormalizedLevenshtein", NormalizedLevenshtein()),
        ("LongestCommonSubsequence", LongestCommonSubsequence()),
    ]

    aggregations = [
        ("min", lambda x: np.min(x)),
        ("max", lambda x: np.max(x)),
        ("mean", lambda x: np.mean(x)),
        ("std", lambda x: np.std(x)),
    ]

    for training_string in training_set:
        for metric_name, metric in metrics:
            distances[metric_name].append(metric.distance(training_string, query_string))

    aggregated_distances = {}

    for k, v in distances.items():
        for agg_name, agg_func in aggregations:
            aggregated_distances[f"{k}_{agg_name}"] = agg_func(v)

    return aggregated_distances


def get_num_monomer(string: str, monomer: str) -> int:
    """Get the amount of a monomer in a polymer string.

    Args:
        string (str): Polymer string
        monomer (str): Monomer

    Returns:
        int: Number of monomers

    Example:
        >>> get_num_monomer("W-A-B-W-W-A-A-A-R-W-B-B-R-R-B-R", "R")
        4
    """
    num = re.findall(f"([\d]+) {monomer}", string)
    try:
        num = int(num[0])
    except Exception:
        num = 0
    return num


def get_polymer_prompt_compostion(prompt: str) -> dict:
    """Get the composition of a polymer prompt.

    Args:
        prompt (str): Polymer prompt

    Returns:
        dict: The composition of the prompt

    Example:
        >>> get_polymer_prompt_compostion("W-A-B-W-W-A-A-A-R-W-B-B-R-R-B-R")
        {'W': 6, 'A': 4, 'B': 4, 'R': 4}
    """
    composition = {}

    for monomer in ["R", "W", "A", "B"]:
        composition[monomer] = get_num_monomer(prompt, monomer)

    return composition


def is_valid_polymer(string):
    """Check if a polymer string is valid.

    Args:
        string (str): Polymer string

    Returns:
        bool: True if valid, False otherwise

    Example:
        >>> is_valid_polymer("W-A-B-W-W-A-A-A-R-W-B-B-R-R-B-R")
        True
    """
    parts = string.split("-")
    valid = False
    for part in parts:
        if part in ["W", "A", "B", "R"]:
            valid = True
        else:
            valid = False
            break
    return valid


def get_continuos_binned_distance(prediction, bin, bins):
    """For inverse design with categories in prompt we compute the distance to the nearest bin edge."""
    in_bin = (prediction >= bins[bin][0]) & (prediction <= bins[bin][1])
    if in_bin:
        loss = 0
    else:
        # compute the minimum distance to bin
        left_edge_distance = abs(prediction - bins[bin][0])
        right_edge_distance = abs(prediction - bins[bin][1])
        loss = min(left_edge_distance, right_edge_distance)
    return loss


def get_inverse_polymer_metrics(
    generated_polymers: Collection[str],
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    max_train: int = 500,
    bins: Optional[List[Tuple[float]]] = None,
) -> dict:
    """Get the inverse design metrics for a set of generated polymers."""
    performances = []

    train_polymers = df_train["label"].tolist()
    representations = [v[0] for v in df_test["representation"].values]

    valid_polymers = []
    composition_mismatches = []
    performance_difference = []
    valid_indices = []
    string_distances_collection = []
    mapes = []
    in_train = []
    for i, polymer in enumerate(generated_polymers):
        try:
            if not is_valid_polymer(polymer):
                continue
            perf = polymer_string2performance(polymer)
            performances.append(perf)
            comp = get_polymer_prompt_compostion(polymer)

            comp_mismatch = composition_mismatch(
                get_polymer_prompt_compostion(df_test["prompt"].iloc[i]), comp
            )
            composition_mismatches.append(comp_mismatch)
            if bins is not None:
                distance = get_continuos_binned_distance(
                    perf["prediction"][0], representations[i], bins
                )
                performance_difference.append(distance)
                mapes.append(distance / np.abs(representations[i]))
            else:
                distance = np.abs(perf["prediction"][0] - representations[i])
                performance_difference.append(distance)
                mapes.append(distance / np.abs(representations[i]))

            valid_polymers.append(polymer)
            valid_indices.append(i)
            string_mismatch = string_distances(train_polymers, polymer)
            if polymer in train_polymers:
                in_train.append(polymer)
            string_distances_collection.append(string_mismatch)
        except Exception:
            pass

    string_distances_collection = pd.DataFrame(string_distances_collection)

    kldiv = PolymerKLDivBenchmark(
        featurize_many_polymers([polymer_convert2smiles(p) for p in train_polymers]),
        min(len(valid_polymers), len(train_polymers)),
    )
    kldiv_score = kldiv.score(valid_polymers)
    novel_smiles_fraction = 1 - len(set(valid_polymers) & set(train_polymers)) / len(valid_polymers)

    return {
        "composition_mismatches": pd.DataFrame(composition_mismatches),
        "summary_composition_mismatches": pd.DataFrame(composition_mismatches).mean().to_dict(),
        "losses": performance_difference,
        "mapes": mapes,
        "kldiv_score": kldiv_score,
        "valid_smiles_fraction": len(valid_polymers) / len(generated_polymers),
        "valid_indices": valid_indices,
        "valid_polymers": valid_polymers,
        "unique_smiles_fraction": len(set(valid_polymers)) / len(valid_polymers),
        "novel_smiles_fraction": novel_smiles_fraction,
        "generated_sequences": generated_polymers,
        "predictions": performances,
        "fraction_in_train": len(in_train) / len(valid_polymers),
        "string_distances_collection": string_distances_collection,
        "string_distances_collection_summary": string_distances_collection.mean().to_dict(),
        "expected_performance": representations,
    }


def get_kappa_intersections(
    index,
    values,
):
    # from the fitted learning curve, find when kappa >0
    # >0.2,, >0.4, >0.6, >0.8

    intersections = {}

    for i in [0, 0.2, 0.4, 0.6, 0.8]:
        intersections[i] = find_learning_curve_intersection(
            i,
            fit_learning_curve(
                index,
                values,
            )[0],
        )

    return intersections


colors = {
    0: plt.cm.RdBu_r(0),
    0.2: plt.cm.RdBu_r(0.2),
    0.4: plt.cm.RdBu_r(0.4),
    0.6: plt.cm.RdBu_r(0.6),
    0.8: plt.cm.RdBu_r(0.8),
}


def add_kappa_vlines(index, values, low=10, high=100, ymax=1.6, ymin=0.2):
    intersections = get_kappa_intersections(index, values)

    for k, v in intersections.items():
        if (v > low) & (v < high):
            plt.vlines(v, ymin, ymax, color=colors[k], alpha=0.8)
            plt.text(
                v + 1, (ymax - ymin) / 2, k, color=colors[k], fontsize=8, rotation=90, alpha=0.8
            )

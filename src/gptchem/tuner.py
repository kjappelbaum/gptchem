import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import openai
import pandas as pd
from fastcore.basics import basic_repr
from fastcore.xtras import dumps
from loguru import logger
from openai import FineTune
from openai.cli import FineTune as FineTuneCli

from .types import PathType
from .utils import make_outdir


def _check_ft_state(ft_id):
    ft = FineTune.retrieve(id=ft_id)
    return ft.get("status")


def get_ft_model_name(ft_id, sleep=60):
    while True:
        ft = FineTune.retrieve(id=ft_id)
        status = ft.get("status")
        logger.debug(f"Fine tuning status: {status}")
        if status == "succeeded":
            return ft.get("fine_tuned_model")
        elif status == "failed":
            raise RuntimeError(f"Fine tuning failed: {ft}")
        time.sleep(sleep)


_PRESETS = {
    "ada-classification": {
        "base_model": "ada",
        "n_epochs": 4,
    },
    "ada-inverse": {
        "base_model": "ada",
        "n_epochs": 2,
    },
}


class Tuner:
    """Wrapper around the OpenAI API for fine tuning."""

    _sleep = 120

    def __init__(
        self,
        base_model: str = "ada",
        batch_size: Optional[int] = None,
        n_epochs: int = 4,
        learning_rate_multiplier: Optional[float] = None,
        outdir: Optional[PathType] = None,
        run_name: str = None,
        wandb_sync: bool = True,
    ) -> None:
        """Initialize a Tuner.

        Args:
            base_model: The base model to fine tune.
                Defaults to "ada".
            batch_size: The batch size to use for fine tuning.
                Defaults to None.
            n_epochs: The number of epochs to fine tune for.
                Defaults to 4.
            learning_rate_multiplier: The learning rate multiplier to use for fine tuning.
                The OpenAI docs state "We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results."
                Defaults to None.
            outdir: The directory to save the fine tuning results to.
                If not specified, a directory will be created in `BASE_OUTDIR`
            run_name: The name of the run. This is used to create the output directory.
            wandb_sync: Whether to sync the results to Weights & Biases.
        """
        self.base_model = base_model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate_multiplier = learning_rate_multiplier
        self.run_name = run_name
        self.wandb_sync = wandb_sync
        self.outdir = (
            outdir if outdir is not None and Path(outdir).exists() else make_outdir(self.run_name)
        )
        self._modelname = None
        self._ft_id = None
        self._train_filename = None
        self._valid_filename = None
        self._train_file_id = None
        self._valid_file_id = None
        self._res = None

    @classmethod
    def from_preset(cls, preset: str = "ada-classification"):
        if preset not in _PRESETS:
            raise ValueError(
                f"Invalid preset: {preset}. Valid presets are: {list(_PRESETS.keys())}"
            )
        return cls(**_PRESETS[preset])

    @property
    def model_name(self):
        if self._modelname is None:
            raise ValueError("Model name not set. Please call `tuner.tune()` first.")

    @property
    def summary(self) -> dict:
        return {
            "base_model": self.base_model,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "run_name": self.run_name,
            "wandb_sync": self.wandb_sync,
            "outdir": str(self.outdir),
            "train_filename": str(self._train_filename),
            "valid_filename": str(self._valid_filename),
            "model_name": self._modelname,
            "ft_id": self._ft_id,
            "date": time.strftime("%Y%m%d_%H%M%S"),
            "train_file_id": self._train_file_id,
            "valid_file_id": self._valid_file_id,
        }

    def _write_file(self, df: pd.DataFrame, data_type: str) -> None:
        """Write a dataframe to a file as json in records form."""
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            if data_type not in ["train", "valid"]:
                raise ValueError(f"Invalid type: {data_type}. Valid types are: ['train', 'valid']")

            filename = os.path.abspath(os.path.join(self.outdir, f"{data_type}.jsonl"))
            df.to_json(filename, orient="records", lines=True, force_ascii=False)
            if data_type == "train":
                self._train_filename = filename
            elif data_type == "valid":
                self._valid_filename = filename

            return filename
        return None

    def tune(self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None) -> dict:
        """Fine tune a model on a dataset.

        Args:
            train_df (pd.DataFrame): Training dataset.
            validation_df (pd.DataFrame, optional): Validation dataset. Defaults to None.

        Returns:
            dict: Summary of the fine tuning run.

        Raises:
            ValueError: If no training dataset is provided.
        """
        if train_df is None:
            raise ValueError("Please provide a training dataset.")
        train_file = self._write_file(train_df, "train")
        valid_file = self._write_file(validation_df, "valid")

        file_args = {
            "training_file": FineTuneCli._get_or_upload(train_file, check_if_file_exists=False)
        }
        self._train_file_id = file_args["training_file"]
        if valid_file is not None:
            file_args["validation_file"] = FineTuneCli._get_or_upload(
                valid_file, check_if_file_exists=False
            )
            self._valid_file_id = file_args["validation_file"]

        settings = {}
        if self.batch_size is not None:
            settings["batch_size"] = self.batch_size
        if self.n_epochs is not None:
            settings["n_epochs"] = self.n_epochs
        if self.learning_rate_multiplier is not None:
            settings["learning_rate_multiplier"] = self.learning_rate_multiplier

        result = openai.FineTune.create(
            **file_args,
            model=self.base_model,
            **settings,
        )
        self._res = result
        logger.debug(f"Requested fine tuning. {result}")
        modelname = None
        try:
            ft_id = result["id"]
            modelname = get_ft_model_name(ft_id, self._sleep)
            # sync runs with wandb
            if self.wandb_sync:
                subprocess.run("openai wandb sync -n 1", shell=True)
        except Exception:
            logger.exception("Fine tuning failed.")

        if modelname is None:
            raise ValueError(f"Fine tuning failed. Result: {result}.")
        self._modelname = modelname
        self._ft_id = ft_id

        with open(os.path.join(self.outdir, "summary.json"), "w") as f:
            f.write(dumps(self.summary))

        logger.debug(f"Fine tuning completed. {self.summary}")
        return self.summary

    def __call__(
        self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None
    ) -> dict:
        return self.tune(train_df, validation_df)

    __repr__ = basic_repr("base_model,batch_size,n_epochs,learning_rate_multiplier,run_name")

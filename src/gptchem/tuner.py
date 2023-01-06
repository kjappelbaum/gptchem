from pyrate_limiter import Duration, Limiter, RequestRate
import openai
import os 
from typing import Optional
import pandas as pd
import subprocess
limiter = Limiter(RequestRate(23, Duration.MINUTE))
import re 
import subprocess
import time 
from pathlib import Path
from openai import FineTune
from .utils import make_outdir
from loguru import logger

openai.api_key = os.getenv('OPENAI_API_KEY')


def _check_ft_state(ft_id):
    ft = FineTune.retrieve(id=ft_id)
    return ft.get('status')

def get_ft_model_name(ft_id, sleep=60):
    while True:
        ft = FineTune.retrieve(id=ft_id)
        if ft.get('status') == 'succeeded':
            return ft.get('fine_tuned_model')
        time.sleep(sleep)
        
_PRESETS = {
    'ada-classification': {
        'base_model': 'ada',
        'n_epochs': 4,
    },
    'ada-inverse': {
        'base_model': 'ada',
        'n_epochs': 2,
    }
}

class Tuner: 
    """Wrapper around the OpenAI API for fine tuning."""
    def __init__(self, base_model: str = 'ada', batch_size: Optional[int] = None, n_epochs: int = 4, learning_rate_multiplier: Optional[float]=None, sleep: int=120, outdir: Optional[PathType] = None, run_name: str = None, wandb_sync: bool = True) -> None:
        self.base_model = base_model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate_multiplier = learning_rate_multiplier
        self.sleep = sleep
        self.run_name = run_name 
        self.wandb_sync = wandb_sync
        self.outdir = outdir if outdir is not None and Path(outdir).exists() else make_outdir(self.run_name)
        self._modelname = None
        self._ft_id = None
        self._train_filename = None
        self._valid_filename = None

    @classmethod
    def from_preset(cls, preset: str = 'ada-classification'): 
        if preset not in _PRESETS: 
            raise ValueError(f"Invalid preset: {preset}. Valid presets are: {list(_PRESETS.keys())}")
        return cls(**_PRESETS[preset])

    @property 
    def model_name(self): 
        if self._modelname is None: 
            raise ValueError("Model name not set. Please call `tuner.tune()` first.")

    @property 
    def summary(self) -> dict: 
        return {
            'base_model': self.base_model,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'learning_rate_multiplier': self.learning_rate_multiplier,
            'run_name': self.run_name,
            'wandb_sync': self.wandb_sync,
            'outdir': self.outdir,
            'train_filename': self._train_filename,
            'valid_filename': self._valid_filename,
            'model_name': self._modelname,
            'ft_id': self._ft_id,
            'date': time.strftime('%Y%m%d_%H%M%S'),
        }

    def _write_file(self, df: pd.DataFrame, data_type: str) -> None: 
        """Write a dataframe to a file as json in records form."""
        if df:
            if data_type not in ['train', 'valid']:
                raise ValueError(f"Invalid type: {data_type}. Valid types are: ['train', 'valid']")
        
            filename = os.path.join(self.outdir, f"{data_type}.jsonl")
            df.to_json(filename, orient='records', lines=True)
            if data_type == 'train':
                self._train_filename = filename
            elif data_type == 'valid':
                self._valid_filename = filename

            return filename
        return None

    def tune(self, train_df: pd.DataFrame, validation_df: pd.DataFrame) -> dict: 
        """Fine tune a model on a dataset."""
        if train_df is None:
            raise ValueError("Please provide a training dataset.")
        train_file = self._write_file(train_df, 'train')
        valid_file = self._write_file(validation_df, 'valid')

        command =  f"openai api fine_tunes.create -t {train_file}  -m {self.base_model}" 
        if self.batch_size is not None: 
            command += f" --batch_size {self.batch_size}"
        if self.learning_rate_multiplier is not None:
            command += f" --learning_rate_multiplier {self.learning_rate_multiplier}"
        if self.n_epochs is not None:
            command += f" --n_epochs {self.n_epochs}"
        if valid_file is not None:
            command += f" -v {valid_file}"

        result = subprocess.run(command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            ft_id = re.findall(r"Created fine-tune: ([\w\d:-]+)\n", result.stdout)[
                    0
                ]
            modelname =  get_ft_model_name(ft_id, self.sleep)
            # sync runs with wandb
            if self.wandb_sync:
                subprocess.run("openai wandb sync -n 1", shell=True)
        except Exception:
            logger.exception(result.stdout, result.stderr)
        
        self._modelname = modelname
        self._ft_id = ft_id
        return self.summary
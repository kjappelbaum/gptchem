import os
import time
from typing import List, Optional

import openai
import pandas as pd
from fastcore.basics import basic_repr, chunked
from pyrate_limiter import Duration, Limiter, RequestRate

_PRESETS = {
    "classification": {
        "max_tokens": 2,  # first token is whitespace
    },
    "inverse": {
        "max_tokens": 800,
    },
    "regression": {
        "max_tokens": 5,
    },
}

limiter = Limiter(RequestRate(23, Duration.MINUTE))


class Querier:
    """Wrapper around the OpenAI API for querying a model for completions.

    This class tries to be as efficient as possible by querying the API
    in batches. It also handles the rate limiting of the API.

    Example:
        >>> querier = Querier("ada")
        >>> df = pd.DataFrame({"prompt": ["This is a test", "This is another test"]})
        >>> completions = querier.query(df)
        >>> assert len(completions) == 2
        True
        >>> assert all([isinstance(c, str) for c in completions])
        True
    """

    _parallel_max = 20
    _sleep = 5
    _stop = "@@@"

    def __init__(self, modelname, max_tokens: int = 10):
        self.modelname = modelname
        self.max_tokens = max_tokens

    @classmethod
    def from_preset(cls, modelname: str, preset: str = "classification"):
        """Factory method to create a Querier from a preset.

        These presets set the max_tokens parameter to a value that is
        appropriate for the task.
        """
        if preset not in _PRESETS:
            raise ValueError(
                f"Invalid preset: {preset}. Valid presets are: {list(_PRESETS.keys())}"
            )
        return cls(modelname, **_PRESETS[preset])

    def query(
        self, df: pd.DataFrame, temperature: float = 0, logprobs: Optional[int] = None
    ) -> dict:
        """Query the model for completions.

        Args:
            df (pd.DataFrame): DataFrame containing a column named "prompt"
            temperature (float): Temperature of the softmax. Defaults to 0.
            logprobs (Optional[int]): The number of logprobs to return.
                For classification, set it to the number of classes.
                Defaults to None.

        Raises:
            ValueError: If df is not a pandas DataFrame
            ValueError: If df does not have a column named "prompt"
            AssertionError: If temperature is < 0

        Returns:
            dict: Dictionary containing the completions and logprobs
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if "prompt" not in df.columns:
            raise ValueError("df must have a column named 'prompt'")
        assert temperature >= 0, "temperature must be >= 0"

        completions = []

        settings = {}
        if logprobs is not None and isinstance(logprobs, int):
            settings["logprobs"] = logprobs

        for chunk in chunked(df["prompt"], self._parallel_max):
            while True:
                try:
                    with limiter.ratelimit("codex", delay=True):
                        completions_ = openai.Completion.create(
                            model=self.modelname,
                            prompt=chunk,
                            temperature=temperature,
                            max_tokens=self.max_tokens,
                            stop=self._stop,
                            **settings,
                        )
                        completions.append(completions_)
                    break
                except openai.error.RateLimitError:
                    time.sleep(self._sleep)
                    continue

        completions = {
            "choices": [choice["text"] for c in completions for choice in c["choices"]],
            "logprobs": [choice["logprobs"] for c in completions for choice in c["choices"]],
            "model": self.modelname,
        }

        return completions

    def __call__(
        self, df: pd.DataFrame, temperature: float = 0, logprobs: Optional[int] = None
    ) -> dict:
        return self.query(df, temperature, logprobs)

    __repr__ = basic_repr("modelname,max_tokens")

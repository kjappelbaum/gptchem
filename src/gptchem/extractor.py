import re
from typing import Union

from fastcore.basics import basic_repr
from fastcore.foundation import L


class BaseExtractor:
    _stop_sequence = "@@"

    def floatify(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return None
        except TypeError:
            return None

    def intify(self, value: str) -> int:
        try:
            return int(self.floatify(value))
        except ValueError:
            return None
        except TypeError:
            return None

    def split(self, value: str) -> str:
        try:
            return value.split(self._stop_sequence)[0]
        except IndexError:
            return None

    def extract(self, data, **kwargs) -> Union[str, float, int]:
        raise NotImplementedError

    def extract_many(self, data, **kwargs) -> L:
        return L([self.extract(entry, **kwargs) for entry in data])

    def extract_many_from_dict(self, data, key="choices", **kwargs) -> L:
        return L([self.extract(entry, **kwargs) for entry in data[key]])

    def __call__(self, data, key="choices", **kwargs):
        return self.extract_many_from_dict(data, key=key, **kwargs)

    __repr__ = basic_repr()


class ClassificationExtractor(BaseExtractor):
    """Extract integers from completions of classification tasks."""

    def extract(self, data, **kwargs) -> int:
        return self.intify(self.split(data).strip())


class FewShotClassificationExtractor(BaseExtractor):
    """Extract integers from completions of few-shot classification tasks."""

    _FIRST_NUMBER_REGEX = re.compile(r"(\d+)")

    def extract(self, data, **kwargs) -> int:
        first_number = self._FIRST_NUMBER_REGEX.findall(data)
        if first_number:
            return self.intify(first_number[0])
        return None


class FewShotRegressionExtractor(BaseExtractor):
    """Extract floats from completions of few-shot regression tasks."""

    _FIRST_NUMBER_REGEX = re.compile(r"(\d+\.\d+)|(\d+)")

    def extract(self, data, **kwargs) -> int:
        first_number = self._FIRST_NUMBER_REGEX.findall(data)
        if first_number:
            return self.floatify(first_number[0][0] or first_number[0][1])
        return None


class RegressionExtractor(BaseExtractor):
    """Extract floats from completions of regression tasks."""

    def extract(self, data, **kwargs) -> float:
        return self.floatify(self.split(data).strip())


class InverseExtractor(BaseExtractor):
    """Extract strings from completions of inverse tasks."""

    def extract(self, data, **kwargs) -> float:
        return self.split(data).split()[0].strip()


class SolventExtractor(BaseExtractor):
    """Extract solvent name and composition from completions of solvent tasks."""

    _SOLVENT_REGEX = re.compile(r"(\d+\.\d+)(\s[\w\(\)=\@]+)")

    def _find_solvent(self, data):
        parts = self._SOLVENT_REGEX.findall(data)

        solvents = {}
        if parts:
            for am, s in parts:
                solvents[s.strip()] = float(am)
            return solvents
        return None

    def extract(self, data, **kwargs) -> dict:
        return self._find_solvent(self.split(data))

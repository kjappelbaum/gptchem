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


class RegressionExtractor(BaseExtractor):
    """Extract floats from completions of regression tasks."""

    def extract(self, data, **kwargs) -> float:
        return self.floatify(self.split(data).strip())

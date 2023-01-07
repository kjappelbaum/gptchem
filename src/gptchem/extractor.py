from typing import Union


class BaseExtractor:

    _stop_sequence = "@@@"

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

    def extract_many(self, data, **kwargs):
        return [self.extract(entry, **kwargs) for entry in data]

    def extract_many_from_dict(self, data, key='choices', **kwargs):
        return [self.extract(entry[key][0], **kwargs) for entry in data]
    

class ClassificationExtractor(BaseExtractor):
    def extract(self, data, **kwargs) -> int:
        return self.intify(self.split(data))


class RegressionExtractor(BaseExtractor):
    def extract(self, data, **kwargs) -> float:
        return self.floatify(self.split(data))

from typing import Union


class BaseExtractor:

    _stop_sequence = "@@@"

    def floatify(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return None

    def intify(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
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

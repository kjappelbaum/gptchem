from gptchem.extractor import BaseExtractor, ClassificationExtractor, RegressionExtractor


def test_base_extractor():
    extractor = BaseExtractor()
    assert extractor.floatify("1.0") == 1.0
    assert extractor.floatify("1") == 1.0

    assert extractor.intify("1.0") == 1
    assert extractor.intify("1") == 1

    assert extractor.split("1.0@@@") == "1.0"


def test_classification_extractor():
    extractor = ClassificationExtractor()
    assert extractor.extract("1.0@@@") == 1

    assert extractor.extract_many(["1.0@@@", "2.0@@@"]) == [1, 2]

    assert extractor.extract_many(["1.0", "2.0@@@", "3.0@@@", "aaa"]) == [1, 2, 3, None]

    assert extractor.extract_many_from_dict([{"choices": ["1.0@@@"]}, {"choices": ["2.0@@@"]}, {"choices": ["3.0@@@"]}, {"choices": ["aaa"]}], key="choices") == [1, 2, 3, None]

def test_regression_extractor():
    extractor = RegressionExtractor()
    assert extractor.extract("1.0@@@") == 1.0

    assert extractor.extract_many(["1.0@@@", "2.0@@@"]) == [1.0, 2.0]

    assert extractor.extract_many(["1.0", "2.0@@@", "3.0@@@", "aaa"]) == [1.0, 2.0, 3.0, None]

    assert extractor.extract_many_from_dict([{"choices": ["1.0@@@"]}, {"choices": ["2.0@@@"]}, {"choices": ["3.0@@@"]}, {"choices": ["aaa"]}], key="choices") == [1.0, 2.0, 3.0, None]
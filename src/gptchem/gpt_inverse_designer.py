from gptchem.gpt_classifier import GPTClassifier
from typing import Union, Optional, List
from gptchem.extractor import InverseExtractor
from gptchem.formatter import InverseDesignFormatter
from gptchem.tuner import Tuner


class GPTInverseDesigner(GPTClassifier):
    def __init__(
        self,
        property_name: Union[str, List[str]],
        tuner: Optional[Tuner] = None,
        querier_settings: Optional[dict] = None,
        save_valid_file: bool = False,
        bias_token: bool = True,
        class_weights: Optional[dict] = None,
    ):
        """Initialize a GPTClassifier.

        Args:
            property_name (Union[str, List[str]]): Name of the property to be predicted.
               This will be part of the prompt.
            tuner (Tuner): Tuner object to be used for fine tuning.
               This specifies the model to be used and the fine-tuning settings.
               Defaults to None. If None, a default tuner will be used.
               This default Tuner will use the `ada` model.
            querier_settings (Optional[dict], optional): Settings for the querier.
                Defaults to None.
            save_valid_file (bool, optional): Whether to save the validation file.
                Defaults to False.
            bias_tokens (bool, optional): Whether to add bias to tokens
                to ensure that only the relevant tokens are generated.
                Defaults to True.
            class_weights (Optional[dict], optional): Class weights to be used for inference.
                Defaults to None. If None, classes will be weighted equally.
                Ensure that the weights add up to 1.
        """
        self.property_name = property_name
        self.tuner = tuner if tuner is not None else Tuner()
        self.querier_setting = (
            querier_settings if querier_settings is not None else {"max_tokens": 3}
        )

        self.extractor = InverseExtractor()
        self.formatter = InverseDesignFormatter(
            representation_column="repr",
            property_columns=["prop"],
            property_names=[property_name],
            num_classes=None,
        )

        self.model_name = None
        self.tune_res = None
        self.save_valid_file = save_valid_file
        self.bias_token = bias_token
        self._input_shape = None
        self._class_weights = class_weights

""" Custom Feature Extractor for Speech2Text model """

from typing import Union

import numpy as np
import torch
from transformers import BatchFeature, PretrainedConfig, Speech2TextFeatureExtractor


class CustomFeatureExtractorConfig(PretrainedConfig):
    model_type = "custom_feature_extractor"


class CustomFeatureExtractor(Speech2TextFeatureExtractor):
    """Custom Feature Extractor for Speech2Text model"""

    def __init__(
        self,
        feature_size=80,
        norm_type="utterance",
        do_ceptral_normalize=True,
        update_norms=True,
        global_means=None,
        global_stds=None,
        *args,
        **kwargs,
    ):
        if norm_type not in ["utterance", "global"]:
            raise ValueError(f"norm_type should be either 'utterance' or 'global'. Got {norm_type}")
        super().__init__(
            feature_size=feature_size,
            do_ceptral_normalize=do_ceptral_normalize and norm_type == "utterance",
            *args,
            **kwargs,
        )
        self.norm_type = norm_type
        self.update_norms = update_norms

        if self.norm_type == "global":
            self.global_means = np.array(
                torch.load(global_means).to_list() if isinstance(global_means, str) else global_means
            )
            self.global_stds = np.array(
                torch.load(global_stds).tolist() if isinstance(global_stds, str) else global_stds
            )

    def global_normalize(self, input_features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        outputs = (input_features - self.global_means) / self.global_stds
        return outputs

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> BatchFeature:
        # Extract features from the example
        batch = super().__call__(*args, **kwargs)

        if self.norm_type == "global":
            batch["input_features"] = self.global_normalize(batch.get("input_features"))
        return batch

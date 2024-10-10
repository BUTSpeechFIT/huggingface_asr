import numpy as np
import torch
from transformers import Speech2TextFeatureExtractor, AutoFeatureExtractor
from typing import List, Union
from transformers import BatchFeature, PretrainedConfig


class CustomFeatureExtractorConfig(PretrainedConfig):
    model_type = "custom_feature_extractor"


class CustomFeatureExtractor(Speech2TextFeatureExtractor):
    def __init__(self, feature_size=80, norm_type="utterance", do_ceptral_normalize=True, update_norms=True,
                 global_means=None, global_stds=None, *args, **kwargs):
        assert norm_type in ["utterance", "global"]
        super().__init__(feature_size=feature_size,
                         do_ceptral_normalize=do_ceptral_normalize and norm_type == "utterance", *args, **kwargs)
        self.norm_type = norm_type
        self.update_norms = update_norms

        if self.norm_type == "global":
            self.global_means = torch.load(global_means) if isinstance(global_means, str) else global_means
            self.global_stds = torch.load(global_stds).tolist if isinstance(global_stds, str) else global_stds
            self._global_means = torch.tensor(self.global_means)
            self._global_stds = torch.tensor(self.global_stds)

    def global_normalize(
            self, input_features: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        outputs = (input_features - self._global_means) / self._global_stds
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

#
# AutoFeatureExtractor.register("custom_feature_extractor", CustomFeatureExtractor)
#
c = CustomFeatureExtractor(do_ceptral_normalize=True, norm_type="global", feature_size=80, normalize_vars=True,
                           normalize_means=True, padding_value=0.0, sampling_rate=16000, return_attention_mask=True,
                           padding_side="right", num_mel_bins=80,
                           update_norms=True, global_means="/mnt/scratch/tmp/ipoloka/hf_exp/experiments/test/global_means.pt",
                            global_stds="/mnt/scratch/tmp/ipoloka/hf_exp/experiments/test/global_stds.pt")
c.push_to_hub("fe_mel_80_global_stats_librispeech")

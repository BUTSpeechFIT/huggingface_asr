from transformers import (
    WavLMConfig,
    WavLMModel,
)
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput

from typing import Optional, Tuple, Union
import torch

class WavLMWrapperConfig(WavLMConfig):
    layer_to_extract = None

class WavLMModelWrapper(WavLMModel):
    def __init__(self, config: WavLMWrapperConfig):
        #config.update({ 'attn_implementation': 'flash_attention_2' })
        super().__init__(config)

    def get_encoder(self):
        return self

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        wav_lm_output = super().forward(
            input_values=input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        if self.config.layer_to_extract is None:
            return wav_lm_output
        else:
            _hidden_state = wav_lm_output.hidden_states[self.config.layer_to_extract]
            wav_lm_output.last_hidden_state = _hidden_state
            return wav_lm_output

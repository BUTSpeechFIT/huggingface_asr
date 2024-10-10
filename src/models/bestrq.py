"""
This module implements the BestRQ model https://arxiv.org/abs/2202.01855.
"""
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.linalg import vector_norm
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _HIDDEN_STATES_START_POSITION,
    Wav2Vec2ForPreTrainingOutput,
    Wav2Vec2PreTrainedModel,
)
from transformers.utils import logging

from models.encoders.e_branchformer import (
    Wav2Vec2EBranchformerConfig,
    Wav2Vec2EBranchformerForCTC,
    Wav2Vec2EBranchformerModel,
    Wav2Vec2EBranchformerEncoderLayerWrapper
)

logger = logging.get_logger(__name__)


class BestRQConfig(PretrainedConfig):
    # model_type = "bestrq-ebranchformer"

    def __init__(
            self, best_rq_codebook_size=8192, best_rq_codebook_dim=16, best_rq_num_books=1, best_rq_in_dim=320, **kwargs
    ):
        super().__init__(**kwargs)
        self.best_rq_codebook_size = best_rq_codebook_size
        self.best_rq_codebook_dim = best_rq_codebook_dim
        self.best_rq_num_books = best_rq_num_books
        self.best_rq_in_dim = best_rq_in_dim


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for scale in tensor.shape[2:]:
            receptive_field_size *= scale
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    amplitude = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return tensor.uniform_(-amplitude, amplitude)


class RandomProjectionQuantizer(nn.Module):
    def __init__(self, config: BestRQConfig):
        super().__init__()
        p_init = torch.zeros((config.best_rq_num_books, config.best_rq_in_dim, config.best_rq_codebook_dim))
        self.register_buffer("P", xavier_uniform_(p_init))
        self.register_buffer(
            "CB",
            F.normalize(
                torch.randn(config.best_rq_num_books, config.best_rq_codebook_size, config.best_rq_codebook_dim)
            ),
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.LongTensor:
        hidden_states = F.normalize(hidden_states[:, None, ...] @ self.P)
        return vector_norm((self.CB.unsqueeze(2) - hidden_states.unsqueeze(2)), dim=-1).argmin(dim=2)


class BestRQMask:
    def _mask_hidden_states(
            self,
            hidden_states: torch.FloatTensor,
            mask_time_indices: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            std: float = 0.1,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = hidden_states[mask_time_indices].normal_(mean=0, std=std)
        return hidden_states


class BestRQModel(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.rpq = RandomProjectionQuantizer(config)
        self.classifiers = nn.ModuleList(
            nn.Linear(config.hidden_size, config.best_rq_codebook_size) for _ in range(config.best_rq_num_books)
        )

    def forward(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            mask_time_indices: Optional[torch.BoolTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        targets = self.rpq(input_values.view((*mask_time_indices.shape[:2], -1)))
        targets = targets.masked_fill(~mask_time_indices[:, None, ...], -100)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        probs = torch.stack([classifier(last_hidden_states) for classifier in self.classifiers], dim=1)
        loss = nn.functional.cross_entropy(probs.flatten(0, 1).transpose(1, 2), targets.flatten(0, 1), reduction="sum")
        loss /= probs.size(1)

        if not return_dict:
            if loss is not None:
                return (loss, last_hidden_states, None, None) + outputs[2:]
            return (last_hidden_states, None, None) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=last_hidden_states,
            codevector_perplexity=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=None,
            diversity_loss=None,
        )


class BestRQEBranchformerForPreTrainingConfig(Wav2Vec2EBranchformerConfig, BestRQConfig):
    model_type = "bestrq-ebranchformer"

    def __init__(self,
                 finetune_with_additional_layer=False,
                 finetune_with_layer_mixing=False,
                 freeze_norm_for_finetunning=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.finetune_with_additional_layer = finetune_with_additional_layer
        self.finetune_with_layer_mixing = finetune_with_layer_mixing
        self.freeze_norm_for_finetunning = freeze_norm_for_finetunning


class BestRQEBranchformerModel(BestRQMask, Wav2Vec2EBranchformerModel):
    def __init__(self, config: BestRQEBranchformerForPreTrainingConfig):
        super().__init__(config)
        if hasattr(self, "masked_spec_embed"):
            del self.masked_spec_embed


# pylint: disable=abstract-method
class BestRQEBranchformerForPreTraining(BestRQModel, Wav2Vec2PreTrainedModel):
    config_class = BestRQEBranchformerForPreTrainingConfig

    def __init__(self, config: BestRQEBranchformerForPreTrainingConfig):
        Wav2Vec2PreTrainedModel.__init__(self, config)
        BestRQModel.__init__(self, config)
        self.wav2vec2 = BestRQEBranchformerModel(config)

        # Initialize weights and apply final processing
        self.post_init()


class BestRQEBranchformerForCTC(Wav2Vec2EBranchformerForCTC):
    config_class = BestRQEBranchformerForPreTrainingConfig

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2EBranchformerModel(config)
        self.blank_projection = nn.Linear(config.hidden_size, 1)

        if config.finetune_with_additional_layer:
            self.additional_layer = Wav2Vec2EBranchformerEncoderLayerWrapper(config, self.wav2vec2)

        if config.finetune_with_layer_mixing:
            layer_weights = torch.zeros(config.num_hidden_layers + 1)
            layer_weights[-1] = 1.0
            self.per_layer_weights = nn.Parameter(layer_weights)
        self.post_init()

    def freeze_encoder(self):
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.config.finetune_with_layer_mixing:
            hidden_states = (
                    torch.stack(outputs.hidden_states)
                    * nn.functional.softmax(self.per_layer_weights, dim=-1)[:, None, None, None]
            ).sum(dim=0)
        else:
            hidden_states = outputs.last_hidden_state

        if self.config.finetune_with_additional_layer:
            hidden_states = self.additional_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                input_values=input_values
            )[0]

        hidden_states = self.dropout(hidden_states)

        if output_hidden_states is not True:
            outputs.hidden_states = None

        logits = self.lm_head(hidden_states)
        logits = torch.concatenate((logits, self.blank_projection(hidden_states)), dim=-1)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=logits.shape[-1] - 1,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from models.decoders.multi_head_gpt2 import GPT2LMMultiHeadModel


class GPT2MultiHeadMixingConfig(GPT2Config):
    model_type = "gpt2-multi-head-mixing"

    def __init__(
        self,
        *args,
        head_locations=None,
        head_weights=None,
        tie_additional_weights=False,
        average_logits=False,
        mixing_mode="full",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.head_locations = head_locations
        self.head_weights = head_weights
        self.tie_additional_weights = tie_additional_weights
        self.average_logits = average_logits
        self.mixing_mode = mixing_mode


class GPT2LMMultiHeadModelMixing(GPT2LMMultiHeadModel):
    config_class = GPT2MultiHeadMixingConfig

    def __init__(self, config: GPT2MultiHeadMixingConfig):
        super().__init__(config)
        if config.mixing_mode == "full":
            self.lm_mixing = nn.Linear(len(self.head_weights) * config.vocab_size, config.vocab_size, bias=True)
        self.post_init()
        if config.mixing_mode == "linear":
            self.lm_mixing = nn.Parameter(
                torch.full((len(self.head_weights), config.vocab_size), 1 / len(self.head_weights)), requires_grad=True
            )
        elif config.mixing_mode == "scalar":
            self.lm_mixing = nn.Parameter(
                torch.full((len(self.head_weights),), 1 / len(self.head_weights)), requires_grad=True
            )
        elif config.mixing_mode == "full":
            self.lm_mixing.weight = nn.Parameter(torch.eye(config.vocab_size).repeat(1, len(self.head_weights)) * 0.5)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[2]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        loss = None
        if self.config.mixing_mode == "full" and isinstance(self.lm_mixing, nn.Linear):
            lm_logits = self.lm_mixing(
                torch.concat(
                    [
                        lm_head(hidden_states[index])
                        for index, lm_head in zip([*self.head_locations, -1], [*self.additional_lm_heads, self.lm_head])
                    ],
                    dim=-1,
                )
            )
        elif self.config.mixing_mode in ("linear", "scalar"):
            lm_logits = (
                torch.stack(
                    [
                        lm_head(hidden_states[index])
                        for index, lm_head in zip([*self.head_locations, -1], [*self.additional_lm_heads, self.lm_head])
                    ],
                    dim=-1,
                )
                * self.lm_mixing.T
            ).sum(dim=-1)
        else:
            raise NotImplementedError(f"Mixing mode {self.config.mixing_mode} not implemented.")

        if labels is not None:
            loss_fct = CrossEntropyLoss()

            shift_logits = lm_logits[-1][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

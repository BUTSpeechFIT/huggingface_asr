from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel, SpeechEncoderDecoderModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class LLMASRModel(SpeechEncoderDecoderModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        number_of_prompt_tokens=16,
        freeze_asr=False,
        freeze_llm=False,
        new_token_ids_mapping_inverted=None,
    ):
        super().__init__(config, encoder, decoder)
        self.number_of_prompt_tokens = number_of_prompt_tokens
        self.soft_prompt = nn.Embedding(self.number_of_prompt_tokens + 1, decoder.config.hidden_size)
        self.linear = nn.Linear(decoder.config.hidden_size, decoder.config.hidden_size)
        self.main_input_name = encoder.main_input_name
        self.new_token_ids_mapping_inverted = new_token_ids_mapping_inverted

        # Initialize soft prompt embeddings
        embeds = self.decoder.get_input_embeddings()
        embeds_mean = embeds.weight.mean(dim=0)
        self.soft_prompt.weight.data = embeds_mean.repeat(self.number_of_prompt_tokens + 1, 1)

        if freeze_asr:
            self.freeze_encoder()
        if freeze_llm:
            self.freeze_decoder()

        if hasattr(self, "enc_to_dec_proj"):
            del self.enc_to_dec_proj

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def prepare_decoder_inputs(self, encoder_outputs, labels, decoder_input_ids=None):
        encoder_predictions = encoder_outputs.logits.argmax(dim=-1)
        device = encoder_predictions.device
        b_size = encoder_predictions.shape[0]

        blank_mask = encoder_predictions == self.encoder.config.pad_token_id
        deduplication_mask = torch.hstack(
            [
                torch.ones((encoder_predictions.shape[0], 1), dtype=torch.bool, device=device),
                encoder_predictions[:, 1:] != encoder_predictions[:, :-1],
            ]
        )
        mask = ~blank_mask & deduplication_mask

        soft_prompts = self.soft_prompt(torch.arange(1, self.number_of_prompt_tokens + 1, device=device))
        end_prompt = self.soft_prompt(torch.tensor(0, device=device)).unsqueeze(0)
        att_mask = None

        if labels is not None:
            llm_labels = []
            asr_embeddings = []
            for idx, sequence in enumerate(encoder_predictions):
                labels_wo_pad = labels[idx][labels[idx] != -100][1:]
                asr_sequence_mask = mask[idx]
                asr_sequence = sequence[mask[idx]].to("cpu").apply_(self.new_token_ids_mapping_inverted.get).to(device)
                asr_sequence_embeds = self.linear(encoder_outputs.hidden_states[idx][asr_sequence_mask])
                asr_embeddings.append(asr_sequence_embeds)
                label_sequence = torch.tensor(
                    (
                        [self.decoder.config.bos_token_id]
                        + [self.decoder.config.pad_token_id] * (self.number_of_prompt_tokens + len(asr_sequence) + 1)
                        + labels_wo_pad.tolist()
                        + [self.decoder.config.eos_token_id]
                    ),
                    device=device,
                )
                llm_labels.append(label_sequence)
            llm_labels = torch.nn.utils.rnn.pad_sequence(
                llm_labels, batch_first=True, padding_value=self.decoder.config.pad_token_id
            )
            input_embeds = self.decoder.get_input_embeddings()(llm_labels)
            input_embeds[:, 1 : self.number_of_prompt_tokens + 1] = soft_prompts
            for idx, asr_sequence_embeds in enumerate(asr_embeddings):
                last_asr_embed_position = self.number_of_prompt_tokens + 1 + asr_sequence_embeds.shape[0]
                input_embeds[idx, self.number_of_prompt_tokens + 1 : last_asr_embed_position] = asr_sequence_embeds
                input_embeds[idx, last_asr_embed_position] = end_prompt

            llm_labels[llm_labels == self.decoder.config.pad_token_id] = -100
        else:
            llm_labels = None
            input_embeds = []
            bos_token_embed = self.decoder.get_input_embeddings()(
                torch.tensor(self.decoder.config.bos_token_id, device=device)
            )
            pad_token_embed = self.decoder.get_input_embeddings()(
                torch.tensor(self.decoder.config.pad_token_id, device=device)
            )
            for idx, sequence in enumerate(encoder_predictions):
                asr_sequence_mask = mask[idx]
                asr_sequence_embeds = self.linear(encoder_outputs.hidden_states[idx][asr_sequence_mask])
                input_embeds.append(
                    torch.vstack([bos_token_embed.unsqueeze(0), soft_prompts, asr_sequence_embeds, end_prompt])
                )
            seq_lengths = torch.tensor([embeds.shape[0] for embeds in input_embeds])
            longest_sequence = seq_lengths.max()
            to_left_pad = longest_sequence - seq_lengths
            for idx, embeds in enumerate(input_embeds):
                input_embeds[idx] = torch.vstack(
                    [pad_token_embed.repeat(longest_sequence - embeds.shape[0], 1), embeds]
                )
            att_mask = (torch.arange(longest_sequence).expand(b_size, longest_sequence).T >= to_left_pad).T
            input_embeds = torch.stack(input_embeds)
            if decoder_input_ids is not None:
                if decoder_input_ids.shape[1] > 1:
                    input_embeds = self.decoder.get_input_embeddings()(decoder_input_ids[:, -1:])
                    att_mask = torch.concatenate(
                        (att_mask, torch.ones((b_size, decoder_input_ids.shape[1] - 1), dtype=torch.bool)), dim=1
                    )
        return input_embeds, llm_labels, att_mask

    # pylint: disable=no-member
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if "attention_mask" in model_kwargs:
            model_kwargs.pop("attention_mask")
        if "labels" in model_kwargs:
            labels = model_kwargs.pop("labels")
        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )
        model_kwargs["labels"] = labels
        return model_kwargs

    def forward(
        self,
        inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # Prepare decoder inputs
        decoder_inputs_embeds, labels, decoder_attention_mask = self.prepare_decoder_inputs(
            encoder_outputs, labels, decoder_input_ids
        )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.decoder.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class LearnableBlankLinear(torch.nn.Module):
    def __init__(self, frozen_linear, blank_id):
        super().__init__()
        self.frozen_linear = frozen_linear
        self.blank_id = blank_id
        self.blank_projection = torch.nn.Linear(frozen_linear.in_features, 1)
        self.frozen_linear.weight.requires_grad = False

    def forward(self, x):
        out = self.frozen_linear(x)
        out[..., self.blank_id] = self.blank_projection(x).squeeze(dim=-1)
        return out

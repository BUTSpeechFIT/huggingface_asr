import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    LogitsProcessorList,
    StoppingCriteriaList,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateOutput
from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder import (
    shift_tokens_right,
)

from models.context.memory_cell import MemoryCell
from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoder,
    JointCTCAttentionEncoderDecoderConfig,
    Seq2SeqLMOutputLosses,
)


class ContextManager(nn.Module):
    def __init__(self, layer, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.memory_dim = config.memory_dim
        self.hidden_states = {}
        self.context_vectors = {}
        self.current_conversations = None
        self.global_convs = None
        self.memory = MemoryCell(config)
        self.input_id_lens = None
        self.is_decoding = False
        self.device = None

    def reset_state(self):
        self.hidden_states = {}
        self.context_vectors = {}
        self.current_conversations = None
        self.global_convs = None
        self.input_id_lens = None

    def erase_memory_cells(self, conv_ids):
        for conv_id in conv_ids:
            if conv_id in self.hidden_states:
                del self.hidden_states[conv_id]
            if conv_id in self.context_vectors:
                del self.context_vectors[conv_id]

    def set_current_conversations(self, ids, global_ids):
        self.current_conversations = ids
        self.global_convs = global_ids

    def init_conversations(self, ids, device):
        self.device = device
        for conv_id in list(set(ids)):
            if conv_id not in self.hidden_states:
                self.hidden_states[conv_id] = self.memory.hidden_init.to(device)
            if conv_id not in self.context_vectors:
                self.context_vectors[conv_id] = self.memory.memory_init.to(device)

    def synchronize_states(self):
        current_rank = dist.get_rank()
        source_ranks = torch.tensor(
            [current_rank if conv_id in self.current_conversations else 0 for conv_id in self.global_convs],
            device=self.device,
        )
        dist.all_reduce(source_ranks)
        hidden_lengths = torch.tensor(
            [
                len(self.hidden_states[conv_id]) if conv_id in self.current_conversations else 0
                for conv_id in self.global_convs
            ],
            device=self.device,
        )
        dist.all_reduce(hidden_lengths)
        hidden_states_to_synchronize = torch.zeros(
            (hidden_lengths.sum(), self.memory.hidden_init.size(1)), device=self.device
        )
        context_states_to_synchronize = torch.zeros(
            (self.memory.memory_init.size(0) * len(self.global_convs), self.memory.memory_init.size(1)),
            device=self.device,
        )
        ends = torch.cumsum(hidden_lengths, dim=0)
        starts = ends - hidden_lengths
        for index, (conv_id, source_rank, start, end) in enumerate(zip(self.global_convs, source_ranks, starts, ends)):
            if source_rank == current_rank:
                hidden_states_to_synchronize[start:end, ...] = self.hidden_states[conv_id]
                context_states_to_synchronize[
                    index * self.memory.memory_init.size(0) : (index + 1) * self.memory.memory_init.size(0), ...
                ] = self.context_vectors[conv_id]
        dist.all_reduce(hidden_states_to_synchronize)
        dist.all_reduce(context_states_to_synchronize)

        for index, (conv_id, start, end) in enumerate(zip(self.global_convs, starts, ends)):
            self.hidden_states[conv_id] = hidden_states_to_synchronize[start:end, ...]
            self.context_vectors[conv_id] = context_states_to_synchronize[
                index * self.memory.memory_init.size(0) : (index + 1) * self.memory.memory_init.size(0), ...
            ]

    @staticmethod
    def broadcast_variable_size_tensor(tensor, current_rank, source_rank):
        # Broadcast the size of the tensor
        size = list(tensor.size())
        handle = dist.broadcast_object_list(size, src=source_rank, async_op=True)

        # Allocate a new tensor with the received size
        if current_rank != source_rank:
            tensor = torch.empty(size, dtype=tensor.dtype, device=tensor.device)

        handle.wait()
        # Broadcast the actual tensor data
        handle = dist.broadcast(tensor, src=source_rank, async_op=True)
        handle.wait()

    def save_state(self, hidden_states, memory_states, hidden_lens):
        if not self.is_decoding:
            for conv_id, hidden_state, memory_state, hidden_len in zip(
                self.current_conversations, hidden_states, memory_states, hidden_lens
            ):
                self.hidden_states[conv_id] = hidden_state[:hidden_len].clone().detach()
                self.context_vectors[conv_id] = memory_state.clone().detach()
            if dist.is_available() and dist.is_initialized():
                self.synchronize_states()

    def set_input_id_lens(self, input_id_lens):
        self.input_id_lens = input_id_lens

    def bind_memory(self):
        self.layer.memory = self.memory

        def hook(module, args, kwargs, output):
            prev_hidden_states = [self.hidden_states[conv] for conv in self.current_conversations]
            prev_hidden_lens = torch.tensor([hidden_state.size(0) for hidden_state in prev_hidden_states])
            prev_hidden_states = pad_sequence(prev_hidden_states, batch_first=True)
            prev_memory_state = pad_sequence(
                [self.context_vectors[conv] for conv in self.current_conversations], batch_first=True
            )
            if len(output) == 2:
                hidden_states, att_weights = output

            attention_mask = kwargs["attention_mask"]

            if attention_mask is not None:
                attention_mask_mha = (
                    attention_mask.clone()
                    .squeeze(dim=1)[:, :1, ...]
                    .repeat_interleave(self.memory_dim, dim=1)
                    .repeat_interleave(self.layer.memory.output_attention.num_heads, dim=0)
                    .transpose(1, 2)
                )
                hidden_lens = attention_mask.size(-1) - (attention_mask < 0).sum(dim=-1).squeeze(dim=1)[:, 0]

            else:
                hidden_lens = self.input_id_lens
                attention_mask_mha = -torch.zeros(
                    (len(hidden_lens), hidden_states.size(1), self.memory_dim),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                for index, hidden_len in enumerate(hidden_lens):
                    attention_mask_mha[index, hidden_len:, :] = torch.finfo(hidden_states.dtype).min
                attention_mask_mha = attention_mask_mha.repeat_interleave(
                    self.layer.memory.output_attention.num_heads, dim=0
                )
                # attention_mask_mha = None

            memory_mask = (
                pad_sequence(
                    [
                        -torch.zeros(
                            prev_hidden_lens[index],
                            self.memory_dim,
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        )
                        for index in range(len(self.current_conversations))
                    ],
                    batch_first=True,
                    padding_value=torch.finfo(hidden_states.dtype).min,
                )
                .repeat_interleave(self.layer.memory.update_attention.num_heads, dim=0)
                .transpose(1, 2)
            )
            # print(module)
            # print((memory_mask != 0).reshape(memory_mask.shape[0], -1).all(dim=1).any())
            # print((attention_mask_mha != 0).reshape(attention_mask_mha.shape[0], -1).all(dim=1).any())

            altered_hidden_states, altered_memory_state = module.memory(
                hidden_states,
                prev_hidden_states,
                prev_memory_state,
                attention_mask_mha=attention_mask_mha,
                memory_mask=memory_mask,
            )

            self.save_state(altered_hidden_states, altered_memory_state, hidden_lens)
            return altered_hidden_states, att_weights

        self.layer.register_forward_hook(hook, with_kwargs=True)


class ContextHolder(TrainerCallback, nn.Module):
    def __init__(self, enc_layers_to_attach_memory, dec_layers_to_attach_memory, config):
        super().__init__()
        self.context_blocks = []
        for layer in enc_layers_to_attach_memory:
            config_encoder = config.encoder
            config_encoder.memory_dim = config.enc_memory_dim
            memory_block = ContextManager(layer, config_encoder)
            memory_block.bind_memory()
            self.context_blocks.append(memory_block)
        for layer in dec_layers_to_attach_memory:
            config_decoder = config.encoder
            config_decoder.memory_dim = config.dec_memory_dim
            memory_block = ContextManager(layer, config_decoder)
            memory_block.bind_memory()
            self.context_blocks.append(memory_block)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for block in self.context_blocks:
            block.reset_state()

    def set_conversations(self, global_ids, local_ids, device):
        for block in self.context_blocks:
            block.set_current_conversations(local_ids, global_ids)
            block.init_conversations(global_ids, device)

    def erase_memory_cells(self, conv_ids):
        for block in self.context_blocks:
            block.erase_memory_cells(conv_ids)

    def set_input_id_lens(self, input_id_lens):
        for block in self.context_blocks:
            block.set_input_id_lens(input_id_lens)

    def started_decoding(self):
        for block in self.context_blocks:
            block.is_decoding = True

    def finished_decoding(self):
        for block in self.context_blocks:
            block.is_decoding = False


class JointCTCAttentionEncoderDecoderContextConfig(JointCTCAttentionEncoderDecoderConfig):
    model_type = "joint-ctc-speech-encoder-decoder-context"
    is_composition = True


class JointCTCAttentionEncoderDecoderWithContext(JointCTCAttentionEncoderDecoder):
    config_class = JointCTCAttentionEncoderDecoderContextConfig
    base_model_prefix = "joint-ctc-speech-encoder-decoder-context"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.enc_memory_cells_location is None or self.config.dec_memory_cells_location is None:
            raise ValueError("Memory cells location is not specified")
        if self.config.enc_memory_dim is None or self.config.dec_memory_dim is None:
            raise ValueError("Memory dimension is not specified")
        self.context_manager = ContextHolder(
            enc_layers_to_attach_memory=[
                layer
                for index, layer in enumerate(self.encoder.base_model.encoder.layers)
                if index in self.config.enc_memory_cells_location
            ],
            dec_layers_to_attach_memory=[
                layer
                for index, layer in enumerate(self.decoder.base_model.h)
                if index in self.config.dec_memory_cells_location
            ],
            config=self.config,
        )

        def decoder_forward_hook(module, args, kwargs):
            id_lens = (
                kwargs["input_ids"].ne(self.generation_config.pad_token_id)
                & kwargs["input_ids"].ne(self.generation_config.eos_token_id)
            ).sum(dim=-1)
            self.context_manager.set_input_id_lens(id_lens)

        self.decoder.register_forward_pre_hook(decoder_forward_hook, with_kwargs=True)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args and key != self.config.conv_ids_column_name:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # pylint: disable=E1101
        input_dict = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, use_cache, encoder_outputs, **kwargs
        )
        input_dict[self.config.conv_ids_column_name] = kwargs[self.config.conv_ids_column_name]
        return input_dict

    @staticmethod
    def duplicate_elements(input_list, expand_size):
        result = [element for element in input_list for _ in range(expand_size)]
        return result

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size, is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **model_kwargs
        )
        if model_kwargs.get(self.config.conv_ids_column_name) is not None:
            model_kwargs[self.config.conv_ids_column_name] = self.duplicate_elements(
                model_kwargs[self.config.conv_ids_column_name], expand_size
            )

        return input_ids, model_kwargs

    def get_recording_ids(self, kwargs):
        recording_ids = kwargs.get(self.config.conv_ids_column_name, [])
        if dist.is_available() and dist.is_initialized():
            all_ids = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_ids, recording_ids)
            all_processed_conversations = sorted(list(set([item for local_ids in all_ids for item in local_ids])))
        else:
            all_processed_conversations = recording_ids
        return recording_ids, all_processed_conversations

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
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutputLosses]:
        recording_ids, all_processed_conversations = self.get_recording_ids(kwargs)
        del kwargs[self.config.conv_ids_column_name]
        self.context_manager.set_conversations(all_processed_conversations, recording_ids, device=self.device)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        output = super().forward(
            inputs=inputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            input_values=input_values,
            input_features=input_features,
            return_dict=return_dict,
            **kwargs,
        )
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        recording_ids, all_processed_conversations = self.get_recording_ids(kwargs)
        self.context_manager.set_conversations(all_processed_conversations, recording_ids, device=self.device)
        self.context_manager.started_decoding()
        # pylint: disable=E1101
        output = super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            assistant_model,
            streamer,
            **kwargs,
        )
        self.context_manager.finished_decoding()
        self.context_manager.set_conversations(all_processed_conversations, recording_ids, device=self.device)
        pseudo_labels = output[:, 1:].clone()
        pseudo_labels[pseudo_labels == self.generation_config.pad_token_id] = -100
        kwargs_copy = kwargs.copy()
        kwargs_copy["labels"] = pseudo_labels
        del kwargs_copy["max_length"]
        del kwargs_copy["num_beams"]
        self(inputs, **kwargs_copy)
        return output

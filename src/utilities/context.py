from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sized, Union

import torch
from torch.utils.data import Dataset, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    BatchFeature,
    PretrainedConfig,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    SequenceFeatureExtractor,
    Speech2TextFeatureExtractor,
    SpeechEncoderDecoderModel,
)
from transformers.trainer_utils import has_length
from transformers.training_args import ParallelMode
from transformers.utils import logging

from models.auto_wrappers import CustomAutoModelForCTC
from models.context.ctc_endoder_decoder_context import (
    JointCTCAttentionEncoderDecoderContextConfig,
    JointCTCAttentionEncoderDecoderWithContext,
)
from models.encoders.e_branchformer import (
    Wav2Vec2EBranchformerConfig,
    Wav2Vec2EBranchformerForCTC,
)
from utilities.collators import SpeechCollatorWithPadding
from utilities.model_utils import average_checkpoints
from utilities.training_arguments import ModelArgumentsContext

logger = logging.get_logger("transformers")
AutoConfig.register("joint-ctc-speech-encoder-decoder-context", JointCTCAttentionEncoderDecoderContextConfig)
AutoModelForSpeechSeq2Seq.register(
    JointCTCAttentionEncoderDecoderContextConfig, JointCTCAttentionEncoderDecoderWithContext
)

AutoConfig.register("wav2vec2-ebranchformer", Wav2Vec2EBranchformerConfig)
CustomAutoModelForCTC.register(Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForCTC)


@dataclass
class SpeechCollatorWithPaddingWithRecordingId(SpeechCollatorWithPadding):
    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        batch = super().__call__(features)
        batch["recording_id"] = [feature["recording_id"] for feature in features]

        return batch


class RandomSamplerWithDependency(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        conv_ids: Optional[List[str]] = None,
        turn_idxs: Optional[List[str]] = None,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        super().__init__()
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got " "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer " "value, but got num_samples={}".format(self.num_samples)
            )

        self.conversations = Counter(conv_ids)
        self.dependent_samples = {conv: [] for conv in self.conversations}
        for index, (conv_id, turn_index) in enumerate(zip(conv_ids, turn_idxs)):
            self.dependent_samples[conv_id].append((turn_index, index))
        self.batch_size = batch_size

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        dependent_samples = [
            [tup[1] for tup in sorted(self.dependent_samples[conv], key=lambda x: x[0], reverse=True)]
            for conv in self.dependent_samples.keys()
        ]
        self.initial_weights = torch.tensor(list(self.conversations.values()), dtype=torch.float)

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            raise Exception("Not implemented yet")
        else:
            weights = self.initial_weights
            for _ in range(self.num_samples // self.batch_size):
                selected_convs = torch.multinomial(weights, self.batch_size, generator=generator)
                weights_update = torch.zeros_like(weights)
                weights_update[selected_convs] = 1
                weights -= weights_update
                # return rand element of dataset in case conversation is already empty
                weights = torch.clip(weights, 0)
                yield from [
                    dependent_samples[conv].pop() for conv in selected_convs if len(dependent_samples[conv]) > 0
                ]

    def __len__(self) -> int:
        return self.num_samples


class ContextAwareTrainer(Seq2SeqTrainer):
    def __init__(self, conv_ids_column_name, turn_index_column_name, **kwargs):
        super().__init__(**kwargs)
        self.conv_ids_column_name = conv_ids_column_name
        self.turn_index_column_name = turn_index_column_name

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        conv_ids = self.train_dataset[self.conv_ids_column_name]
        turn_idxs = self.train_dataset[self.turn_index_column_name]

        # Build the sampler.
        if self.args.group_by_length:
            raise NotImplementedError("Not implemented yet")
        else:
            if self.args.world_size <= 1:
                return RandomSamplerWithDependency(
                    self.train_dataset,
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    conv_ids=conv_ids,
                    turn_idxs=turn_idxs,
                    generator=generator,
                )
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                raise NotImplementedError("Not implemented yet")
            else:
                raise NotImplementedError("Not implemented yet")

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        conv_ids = eval_dataset[self.conv_ids_column_name]
        turn_idxs = eval_dataset[self.turn_index_column_name]
        # Deprecated code
        if self.args.use_legacy_prediction_loop:
            raise NotImplementedError("Not implemented yet")

        if self.args.world_size <= 1:
            return RandomSamplerWithDependency(
                eval_dataset, self.args.per_device_eval_batch_size, conv_ids=conv_ids, turn_idxs=turn_idxs
            )
        else:
            raise NotImplementedError("Not implemented yet")


def fetch_config(
    enc_config_path: str, dec_config_path: str, base_config: Dict, config_overrides: str
) -> PretrainedConfig:
    enc_config = AutoConfig.from_pretrained(enc_config_path)
    dec_config = AutoConfig.from_pretrained(dec_config_path)
    config = JointCTCAttentionEncoderDecoderContextConfig.from_encoder_decoder_configs(enc_config, dec_config)
    if config_overrides is not None:
        logger.info(f"Overriding config: {config_overrides}")
        parsed_dict = dict(x.split("=") for x in config_overrides.split(","))
        base_config.update(parsed_dict)
    kwargs_encoder = {
        argument[len("encoder_") :]: value for argument, value in base_config.items() if argument.startswith("encoder_")
    }
    kwargs_decoder = {
        argument[len("decoder_") :]: value
        for argument, value in base_config.items()
        if argument.startswith("decoder_") and argument != "decoder_start_token_id"
    }
    config.encoder.update(kwargs_encoder)
    config.decoder.update(kwargs_decoder)
    config.update(base_config)
    return config


def instantiate_aed_model(
    model_args: ModelArgumentsContext, tokenizer: PreTrainedTokenizer, feature_extractor: SequenceFeatureExtractor
) -> SpeechEncoderDecoderModel:
    base_model_config = {
        "encoder_layerdrop": 0.0,
        "ctc_weight": model_args.ctc_weight,
        "encoder_ctc_loss_reduction": "mean",
        "pad_token_id": tokenizer.pad_token_id,
        "encoder_pad_token_id": tokenizer.pad_token_id,
        "encoder_vocab_size": len(tokenizer),
        "decoder_vocab_size": len(tokenizer),
        "lsm_factor": model_args.lsm_factor,
        "shared_lm_head": model_args.shared_lm_head,
        "encoder_expect_2d_input": model_args.expect_2d_input,
        "decoder_start_token_id": tokenizer.bos_token_id,
        "decoder_pos_emb_fixed": model_args.decoder_pos_emb_fixed,
        "turn_index_column_name": model_args.turn_index_column_name,
        "conv_ids_column_name": model_args.conv_ids_column_name,
        "enc_memory_cells_location": model_args.enc_memory_cells_location,
        "enc_memory_dim": model_args.enc_memory_dim,
        "dec_memory_cells_location": model_args.dec_memory_cells_location,
        "dec_memory_dim": model_args.dec_memory_dim,
    }
    if base_model_config["encoder_expect_2d_input"] and isinstance(feature_extractor, Speech2TextFeatureExtractor):
        base_model_config["encoder_second_dim_input_size"] = feature_extractor.num_mel_bins

    # 4. Initialize seq2seq model
    if model_args.from_pretrained:
        config = AutoConfig.from_pretrained(model_args.from_pretrained)
        config.update(base_model_config)
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints(model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, config=config)
    elif model_args.from_encoder_decoder_config:
        config = fetch_config(
            model_args.base_encoder_model,
            model_args.base_decoder_model,
            base_model_config,
            model_args.config_overrides,
        )
        model = JointCTCAttentionEncoderDecoderWithContext(config=config)
    else:
        model = JointCTCAttentionEncoderDecoderWithContext.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=model_args.base_encoder_model,
            decoder_pretrained_model_name_or_path=model_args.base_decoder_model,
            **base_model_config,
        )
    return model

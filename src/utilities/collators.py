from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from contextlib import contextmanager
import ast

import torch
import json
import re
from transformers import (
    BatchFeature,
    PreTrainedModel,
    PreTrainedTokenizer,
    Speech2TextFeatureExtractor,
    Wav2Vec2FeatureExtractor,
    WhisperFeatureExtractor,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)


@dataclass
class SpeechCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.SequenceFeatureExtractor`)
            The feature extractor used for processing the data.
        tokenizer (:class:`~transformers.PreTrainedTokenizer`)
            The processor used for processing the text data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
        defaults to :obj:`True`):
            Select a strategy to pad the returned sequences
            (according to the model's padding side and padding index) among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    Based upon: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/
                /Fine_tuning_Wav2Vec2_for_English_ASR.ipynb
    """

    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor, WhisperFeatureExtractor]
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: str = None
    text_path: str = None
    model_input_name: str = True
    mask_unks: bool = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature[self.audio_path].squeeze(dim=0)})
            for feature in features
        ]

        if isinstance(features[0][self.text_path], list):
            labels_prime = [ ' '.join(feature[self.text_path]) for feature in features ]
        else:
            labels_prime = [ feature[self.text_path] for feature in features ]

        labels = self.tokenizer.batch_encode_plus(
            labels_prime,
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            batch[self.feature_extractor.model_input_names[0]] = batch[
                self.feature_extractor.model_input_names[0]
            ].transpose(-2, -1)

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)

        if self.mask_unks:
            labels = labels.masked_fill(labels.eq(self.tokenizer.unk_token_id), -100)

        batch["labels"] = labels

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        return batch


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: PreTrainedModel
    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor]
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10
    sampling_rate: Optional[int] = 16_000
    audio_path: str = None
    model_input_name: str = True

    def __post_init__(self):
        if not isinstance(self.feature_extractor, (Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor)):
            raise ValueError(
                f"`feature_extractor` has to be of type {Wav2Vec2FeatureExtractor} or {Speech2TextFeatureExtractor} for {self.__class__}."
            )

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Union[Dict[str, torch.Tensor], BatchFeature]:
        # reformat list to dict and set to pytorch format
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature[self.audio_path].squeeze(dim=0)})
            for feature in features
        ]
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch[self.feature_extractor.model_input_names[0]].device
        batch_size = batch[self.feature_extractor.model_input_names[0]].shape[0]

        input_len = (
            batch[self.feature_extractor.model_input_names[0]].shape[-2]
            if isinstance(self.feature_extractor, Speech2TextFeatureExtractor)
            else batch[self.feature_extractor.model_input_names[0]].shape[-1]
        )
        # pylint: disable=no-member
        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(input_len)
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            # pylint: disable=no-member
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            # pylint: disable=no-member
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        del batch["sub_attention_mask"]
        return batch



@dataclass
class SpeechMTCollatorWithPadding:
    """
    Data collator returning speech, source and target translation text.
    Args:
        feature_extractor (:class:`~transformers.SequenceFeatureExtractor`)
            The feature extractor used for processing the data.
        tokenizer (:class:`~transformers.PreTrainedTokenizer`)
            The processor used for processing the text data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
        defaults to :obj:`True`):
            Select a strategy to pad the returned sequences
            (according to the model's padding side and padding index) among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    Based upon: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/
                /Fine_tuning_Wav2Vec2_for_English_ASR.ipynb
    """

    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor]
    tokenizer_source: PreTrainedTokenizer
    tokenizer_target: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: str = None
    target_text_path: str = None
    source_text_path: str = None
    model_input_name: str = None
    source_prompt_prefix: str = ''

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature[self.audio_path].squeeze(dim=0)})
            for feature in features
        ]

        labels = self.tokenizer_target.batch_encode_plus(
            [feature[self.target_text_path] for feature in features],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )

        source_text_ids = self.tokenizer_source.batch_encode_plus(
            [self.source_prompt_prefix + feature[self.source_text_path] for feature in features],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            batch[self.feature_extractor.model_input_names[0]] = batch[
                self.feature_extractor.model_input_names[0]
            ].transpose(-2, -1)

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels
        batch["mm_input_ids"] = source_text_ids['input_ids']
        batch["mm_attention_mask"] = source_text_ids['attention_mask']

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        return batch

@dataclass
class SpeechAlignedCollatorWithPadding:
    """
    Data collator returning speech, source and target translation text.
    Args:
        feature_extractor (:class:`~transformers.SequenceFeatureExtractor`)
            The feature extractor used for processing the data.
        tokenizer (:class:`~transformers.PreTrainedTokenizer`)
            The processor used for processing the text data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
        defaults to :obj:`True`):
            Select a strategy to pad the returned sequences
            (according to the model's padding side and padding index) among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    Based upon: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/
                /Fine_tuning_Wav2Vec2_for_English_ASR.ipynb
    """

    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor]
    tokenizer_source: Optional[PreTrainedTokenizer] = None
    tokenizer_target: Optional[PreTrainedTokenizer] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    model_input_name: Optional[str] = None
    prompt_prefix: Optional[str] = None
    prompt_suffix: Optional[str] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature[self.audio_path].squeeze(dim=0)})
            for feature in features
        ]

        if isinstance(features[0][self.text_path], list):
            labels_prime = [ ' '.join(feature[self.text_path]) for feature in features ]
        else:
            labels_prime = [ feature[self.text_path] for feature in features ]

        labels = self.tokenizer_target.batch_encode_plus(
            labels_prime,
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )

        # cut off the bos token
        if labels['input_ids'][0, 0] == self.tokenizer_target.bos_token_id:
            labels['input_ids'] = labels['input_ids'][:,1:]
            labels['attention_mask'] = labels['attention_mask'][:,1:]

        if self.prompt_prefix is not None:
            prompt_prefix_ids = self.tokenizer_source.batch_encode_plus(
                [self.prompt_prefix for _ in features],
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

            # cut off the prefix eos token id
            if prompt_prefix_ids['input_ids'][0, -1] == self.tokenizer_source.eos_token_id:
                prompt_prefix_ids['input_ids'] = prompt_prefix_ids['input_ids'][:,:-1]
                prompt_prefix_ids['attention_mask'] = prompt_prefix_ids['attention_mask'][:,:-1]

        else:
            prompt_prefix_ids = None

        if self.prompt_suffix is not None:
            prompt_suffix_ids = self.tokenizer_source.batch_encode_plus(
                [self.prompt_suffix for _ in features],
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

            # cut off the bos token
            if prompt_suffix_ids['input_ids'][0, 0] == self.tokenizer_target.bos_token_id:
                prompt_suffix_ids['input_ids'] = prompt_suffix_ids['input_ids'][:,1:]
                prompt_suffix_ids['attention_mask'] = prompt_suffix_ids['attention_mask'][:,1:]

            # cut off the eos token
            if prompt_suffix_ids['input_ids'][0, -1] == self.tokenizer_target.eos_token_id:
                prompt_suffix_ids['input_ids'] = prompt_suffix_ids['input_ids'][:,:-1]
                prompt_suffix_ids['attention_mask'] = prompt_suffix_ids['attention_mask'][:,:-1]

        else:
            prompt_suffix_ids = None

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            batch[self.feature_extractor.model_input_names[0]] = batch[
                self.feature_extractor.model_input_names[0]
            ].transpose(-2, -1)

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if prompt_prefix_ids is not None:
            batch["prompt_prefix_ids"] = prompt_prefix_ids['input_ids']
            batch["prompt_prefix_mask"] = prompt_prefix_ids['attention_mask']

        if prompt_suffix_ids is not None:
            batch["prompt_suffix_ids"] = prompt_suffix_ids['input_ids']
            batch["prompt_suffix_mask"] = prompt_suffix_ids['attention_mask']

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        return batch

@dataclass
class MultiTokMTCollatorWithPadding:
    """
    Data collator returning speech, source and target translation text.
    Args:
        feature_extractor (:class:`~transformers.SequenceFeatureExtractor`)
            The feature extractor used for processing the data.
        tokenizer (:class:`~transformers.PreTrainedTokenizer`)
            The processor used for processing the text data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
        defaults to :obj:`True`):
            Select a strategy to pad the returned sequences
            (according to the model's padding side and padding index) among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    Based upon: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/
                /Fine_tuning_Wav2Vec2_for_English_ASR.ipynb
    """

    tokenizer_source: Optional[PreTrainedTokenizer] = None
    tokenizer_target: Optional[PreTrainedTokenizer] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: Optional[str] = None
    source_text_path: Optional[str] = None
    target_text_path: Optional[str] = None
    model_input_name: Optional[str] = None
    prompt_prefix: Optional[str] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> dict:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        labels = self.tokenizer_target.batch_encode_plus(
            [feature[self.target_text_path] for feature in features],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )

        input_ids = self.tokenizer_source.batch_encode_plus(
            [feature[self.source_text_path] for feature in features],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )

        batch = {}
        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels
        batch["input_ids"] = input_ids["input_ids"]
        batch["attention_mask"] = input_ids["attention_mask"]

        return batch

@contextmanager
def left_padding(tokenizer):
    original_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = 'left'
        yield
    finally:
        tokenizer.padding_side = original_side

@contextmanager
def nadd_eos(tokenizer):
    original = tokenizer.add_eos_token
    try:
        tokenizer.add_eos_token = False
        yield
    finally:
        tokenizer.add_eos_token = original

@contextmanager
def nadd_bos(tokenizer):
    original = tokenizer.add_bos_token
    try:
        tokenizer.add_bos_token = False
        yield
    finally:
        tokenizer.add_bos_token = original

@dataclass
class GeneralContextCollator:
    """ Data collator for general/fisher ASR dataset with conversation history """

    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor]
    tokenizer: Optional[PreTrainedTokenizer] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    model_input_name: Optional[str] = None
    context_prefix: Optional[str] = 'Context: '
    prompt_prefix: Optional[str] = None
    prompt_suffix: Optional[str] = None
    max_context: Optional[int] = 5
    context_trunc_to_shortest: Optional[bool] = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature[self.audio_path].squeeze(dim=0)})
            for feature in features
        ]

        if isinstance(features[0][self.text_path], list):
            labels_words = [ ' '.join(feature[self.text_path]) for feature in features ]
        else:
            labels_words = [ feature[self.text_path] for feature in features ]

        with nadd_bos(self.tokenizer):
            labels = self.tokenizer.batch_encode_plus(
                labels_words,
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        # 2) Depending on the mode, assemble the context
        max_context = self.max_context
        context_words = []
        for feature in features:
            if feature['context']:
                context_words.append(self.context_prefix + ' '.join([ ' '.join(turn['labels']) for turn in feature['context'][-max_context:]]))
            else:
                context_words.append(self.context_prefix)

        with left_padding(self.tokenizer), nadd_eos(self.tokenizer):
            context = self.tokenizer.batch_encode_plus(
                context_words,
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        # 3) Tokenize the embedding prefix
        if self.prompt_prefix not in [None, '']:
            with nadd_bos(self.tokenizer), nadd_eos(self.tokenizer):
                prompt_prefix_ids = self.tokenizer.batch_encode_plus(
                    [self.prompt_prefix for _ in features],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )
        else:
            prompt_prefix_ids = None

        # 3) Tokenize the embedding suffix
        if self.prompt_suffix not in [None, '']:
            with nadd_eos(self.tokenizer), nadd_bos(self.tokenizer):
                #if self.mode == 'turns': FIXME: so, the initial speaker tag should probably be
                # added to the suffix..
                prompt_suffix_ids = self.tokenizer.batch_encode_plus(
                    [self.prompt_suffix for _ in features],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )
        else:
            prompt_suffix_ids = None

        # 4) Apply padding to the features
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            batch[self.feature_extractor.model_input_names[0]] = batch[
                self.feature_extractor.model_input_names[0]
            ].transpose(-2, -1)

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if context is not None:
            batch["context_ids"] = context['input_ids']
            batch["context_mask"] = context['attention_mask']

        if prompt_prefix_ids is not None:
            batch["prompt_prefix_ids"] = prompt_prefix_ids['input_ids']
            batch["prompt_prefix_mask"] = prompt_prefix_ids['attention_mask']

        if prompt_suffix_ids is not None:
            batch["prompt_suffix_ids"] = prompt_suffix_ids['input_ids']
            batch["prompt_suffix_mask"] = prompt_suffix_ids['attention_mask']

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        return batch

@dataclass
class FisherContextCollatorLeftPadding:
    """ Data collator for the fisher dataset augmented with conversation context. """

    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor]
    tokenizer: Optional[PreTrainedTokenizer] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    model_input_name: Optional[str] = None
    context_prefix: Optional[str] = 'Context: '
    prompt_prefix: Optional[str] = None
    prompt_suffix: Optional[str] = None
    mode: Optional[str] = "default" # available modes: default, turns,
    max_context: Optional[int] = 5
    context_trunc_to_shortest: Optional[bool] = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature[self.audio_path].squeeze(dim=0)})
            for feature in features
        ]

        # 1) Depending on the mode, assemble the labels
        if self.mode == 'default':
            labels_words = [ ' '.join(feature[self.text_path]) for feature in features ]

        elif self.mode == 'turns':
            labels_words = []
            for feature in features:
                last_spk = None # prevent the same speaker label to appear mutliple times in a row
                tmp = []

                # remap the speaker tags to start with A, as there is no context
                if not feature['context']:
                    if feature['speakers'][0] == 'B':
                        for i in range(len(feature['speakers'])):
                            if feature['speakers'][i] == 'B':
                                feature['speakers'][i] = 'A'
                            else:
                                feature['speakers'][i] = 'B'

                for utt, spk in zip(feature[self.text_path], feature['speakers']): 
                    if spk == last_spk:
                        tmp.append(utt.strip())
                    else:
                        last_spk = spk
                        tmp.append(spk + ': ' + utt.strip())

                labels_words.append(' '.join(tmp))

        with nadd_bos(self.tokenizer):
            labels = self.tokenizer.batch_encode_plus(
                labels_words,
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        # 2) Depending on the mode, assemble the context
        max_context = self.max_context
        if self.context_trunc_to_shortest: # FIXME: contexts of length zero break everything...
            # NOTE: for now, I'm leaving the 1 here, but it should probably have a better solution..
            max_context = min(min(list(map(lambda x: len(x['context']) if x['context'] else 1, features))), self.max_context)

        if self.mode == 'default':
            # the most monstrous list comprehension I've ever written..
            context_words = []
            for feature in features:
                # FIXME: context zero lengt
                if feature['context']:
                    context_words.append(self.context_prefix + ' '.join([ ' '.join(turn['labels']) for turn in feature['context'][-max_context:]]))
                else:
                    context_words.append(self.context_prefix)

        elif self.mode == 'turns': # FIXME: zero context kinda makes even less sense in this scenario...
            context_words = []
            for feature in features:
                if feature['context']:
                    tmp = []
                    last_spk = None
                    for turn in feature['context'][-max_context:]:
                        for utt, spk in zip(turn['labels'], turn['speakers']): 
                            if spk == last_spk:
                                tmp.append(utt.strip())
                            else:
                                last_spk = spk
                                tmp.append(spk + ': ' + utt.strip())

                    context_words.append(self.context_prefix + ' '.join(tmp))
                else:
                    context_words.append(self.context_prefix)

        with left_padding(self.tokenizer), nadd_eos(self.tokenizer):
            context = self.tokenizer.batch_encode_plus(
                context_words,
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        # 3) Tokenize the embedding prefix
        if self.prompt_prefix not in [None, '']:
            with nadd_bos(self.tokenizer), nadd_eos(self.tokenizer):
                prompt_prefix_ids = self.tokenizer.batch_encode_plus(
                    [self.prompt_prefix for _ in features],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )
        else:
            prompt_prefix_ids = None

        # 3) Tokenize the embedding suffix
        if self.prompt_suffix not in [None, '']:
            with nadd_eos(self.tokenizer), nadd_bos(self.tokenizer):
                #if self.mode == 'turns': FIXME: so, the initial speaker tag should probably be
                # added to the suffix..
                prompt_suffix_ids = self.tokenizer.batch_encode_plus(
                    [self.prompt_suffix for _ in features],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )
        else:
            prompt_suffix_ids = None

        # 4) Apply padding to the features
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            batch[self.feature_extractor.model_input_names[0]] = batch[
                self.feature_extractor.model_input_names[0]
            ].transpose(-2, -1)

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if context is not None:
            batch["context_ids"] = context['input_ids']
            batch["context_mask"] = context['attention_mask']

        if prompt_prefix_ids is not None:
            batch["prompt_prefix_ids"] = prompt_prefix_ids['input_ids']
            batch["prompt_prefix_mask"] = prompt_prefix_ids['attention_mask']

        if prompt_suffix_ids is not None:
            batch["prompt_suffix_ids"] = prompt_suffix_ids['input_ids']
            batch["prompt_suffix_mask"] = prompt_suffix_ids['attention_mask']

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        return batch

@dataclass
class SlurpCollator:
    """ Data collator for the SLURP dataset for TOD """

    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor]
    tokenizer: Optional[PreTrainedTokenizer] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    model_input_name: Optional[str] = None
    prompt_prefix: Optional[str] = 'Given the following speech utterance, transcribe it and infer the task scenario, action and slots in JSON format: '
    prompt_suffix: Optional[str] = None
    use_slots: Optional[bool] = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature['audio'].squeeze(dim=0)})
            for feature in features
        ]

        labels_text = [ feature[self.text_path] for feature in features ]

        jsons = [ json.loads(feature['full_line']) for feature in features ]
        scenario = [ json['scenario'] for json in jsons ]
        action = [ json['action'] for json in jsons ]
        pattern = pattern = r'\[(.*?) : (.*?)\]'
        if self.use_slots:
            slots = [ json.dumps(dict(re.findall(pattern, j['sentence_annotation']))) for j in jsons ]

        labels_json = []
        for i, label in enumerate(labels_text):
            if self.use_slots:
                labels_json.append(
                    '"' + label + '"'
                    + ', "scenario": "' + scenario[i] + '"' + ', "action": "' + action[i] 
                    + '", "slots": ' + slots[i] + '}'
                )
            else:
                labels_json.append(
                    '"' + label + '"'
                    + ', "scenario": "' + scenario[i] + '"' + ', "action": "' + action[i] + '"}'
                )
        
        with nadd_bos(self.tokenizer):
            labels = self.tokenizer.batch_encode_plus(
                labels_json,
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        embed_suffix = ' {"transcript": '
        embed_prefix = self.prompt_prefix

        # 3) Tokenize the embedding prefix
        if embed_prefix not in [None, '']:
            with nadd_eos(self.tokenizer):
                prompt_prefix_ids = self.tokenizer.batch_encode_plus(
                    [embed_prefix for _ in features],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )
        else:
            prompt_prefix_ids = None

        # 3) Tokenize the embedding suffix
        if embed_suffix not in [None, '']:
            with nadd_eos(self.tokenizer), nadd_bos(self.tokenizer):
                prompt_suffix_ids = self.tokenizer.batch_encode_plus(
                    [embed_suffix for _ in features],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )
        else:
            prompt_suffix_ids = None

        # 4) Apply padding to the features
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            batch[self.feature_extractor.model_input_names[0]] = batch[
                self.feature_extractor.model_input_names[0]
            ].transpose(-2, -1)

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if prompt_prefix_ids is not None:
            batch["prompt_prefix_ids"] = prompt_prefix_ids['input_ids']
            batch["prompt_prefix_mask"] = prompt_prefix_ids['attention_mask']

        if prompt_suffix_ids is not None:
            batch["prompt_suffix_ids"] = prompt_suffix_ids['input_ids']
            batch["prompt_suffix_mask"] = prompt_suffix_ids['attention_mask']

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        return batch

@dataclass
class SpokenWOZCollator:
    """ Data collator for the SpokenWOZ dataset for TOD

    The propmpt structure for this collator is as follows:
        %prompt_prefix% %speech_embeds% %embed_suffix% %generated_text%

        embed_suffix: {"dialogue_history": "%dialogue_history%", "current_turn": 
        generated_text: "%transcript%", "domains": "%domains%", "slots": %slots% }

    """

    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor]
    tokenizer: Optional[PreTrainedTokenizer] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    context_text_path: Optional[str] = None
    model_input_name: Optional[str] = None
    prompt_prefix: Optional[str] = 'Given the following user input speech utterance and the user input history, transcribe the utterance and infer the dialogue state domains and slots in JSON format: '
    prompt_suffix: Optional[str] = None

    def flatten_dict(self, metadata):
        ret = {}
        for key, val in metadata.items():
            try:
                val['book'].pop('booked')
            except:
                pass
            ret[key] = val['book'] | val['semi']
        return ret

    def minimize_dict(self, metadata):
        ret = {}
        for key, val in metadata.items():
            vals = {}
            for kk, vv in val.items():
                if vv != "":
                    vals[kk] = vv
            if vals != {}:
                ret[key] = vals

        return ret

    def get_domains(self, metadata):
        return list(metadata.keys())

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature[self.audio_path].squeeze(dim=0)})
            for feature in features
        ]

        labels_text = [ feature[self.text_path] for feature in features ]
        histories = []
        for feature in features:
            histories.append(' '.join(feature['context'][self.context_text_path])) # TODO is this correct?

        slots = [ self.minimize_dict(self.flatten_dict(json.loads(feature['metadata']))) for feature in features ]
        domains = [ self.get_domains(slot) for slot in slots ]

        labels_json = []
        for i, label in enumerate(labels_text):
            labels_json.append(f'"{label}", "domains": {json.dumps(domains[i])}, "slots": {json.dumps(slots[i])}' + '}')
        
        with nadd_bos(self.tokenizer):
            labels = self.tokenizer.batch_encode_plus(
                labels_json,
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        # 3) Tokenize the embedding prefix
        embed_prefix = self.prompt_prefix
        if embed_prefix not in [None, '']:
            with nadd_eos(self.tokenizer):
                prompt_prefix_ids = self.tokenizer.batch_encode_plus(
                    [embed_prefix for _ in features],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )
        else:
            prompt_prefix_ids = None

        embed_suffix = []
        for history in histories:
            embed_suffix.append(' {' + f'"dialogue_history": "{history}", "current_turn": ')

        # 3) Tokenize the embedding suffix
        with nadd_eos(self.tokenizer), nadd_bos(self.tokenizer), left_padding(self.tokenizer):
            prompt_suffix_ids = self.tokenizer.batch_encode_plus(
                [e for e in embed_suffix],
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        # 4) Apply padding to the features
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            batch[self.feature_extractor.model_input_names[0]] = batch[
                self.feature_extractor.model_input_names[0]
            ].transpose(-2, -1)

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if prompt_prefix_ids is not None:
            batch["prompt_prefix_ids"] = prompt_prefix_ids['input_ids']
            batch["prompt_prefix_mask"] = prompt_prefix_ids['attention_mask']

        if prompt_suffix_ids is not None:
            batch["prompt_suffix_ids"] = prompt_suffix_ids['input_ids']
            batch["prompt_suffix_mask"] = prompt_suffix_ids['attention_mask']

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        return batch

@dataclass
class WOZCollator:
    """ Data collator for the SpokenWOZ and MultiWOZ datasets for TOD

    The propmpt structure for this collator is as follows:
        %prompt_prefix% %speech_embeds% %embed_suffix% %generated_text%

        embed_suffix: {"dialogue_history": "%dialogue_history%", "current_turn": 
        generated_text: "%transcript%", "domains": "%domains%", "slots": %slots% }

    """

    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor]
    tokenizer: Optional[PreTrainedTokenizer] = None
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    model_input_name: Optional[str] = None
    prompt_prefix: Optional[str] = 'Given the following user input speech utterance and the user input history, transcribe the utterance and infer the dialogue state domains and slots in JSON format: '
    prompt_suffix: Optional[str] = None
    use_agent_history: Optional[bool] = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature[self.audio_path].squeeze(dim=0)})
            for feature in features
        ]

        labels_text = [ feature[self.text_path] for feature in features ]
        histories = []
        if self.use_agent_history:
            for feature in features:
                h = []
                for u, a in zip(feature['context']['text'], feature['context']['agent_text']):
                    h.append("USER: " + u + " AGENT: " + a)
                histories.append(' '.join(h))
        else:
            for feature in features:
                histories.append(' '.join(feature['context']['text']))

        slots = [ ast.literal_eval(feature['slots']) for feature in features ]
        domains = [ ast.literal_eval(feature['domains']) for feature in features ]

        labels_json = []
        for i, label in enumerate(labels_text):
            labels_json.append(f'"{label}", "domains": {json.dumps(domains[i])}, "slots": {json.dumps(slots[i])}' + '}')
        
        with nadd_bos(self.tokenizer):
            labels = self.tokenizer.batch_encode_plus(
                labels_json,
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        # 3) Tokenize the embedding prefix
        embed_prefix = self.prompt_prefix
        if embed_prefix not in [None, '']:
            with nadd_eos(self.tokenizer):
                prompt_prefix_ids = self.tokenizer.batch_encode_plus(
                    [embed_prefix for _ in features],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )
        else:
            prompt_prefix_ids = None

        embed_suffix = []
        for history in histories:
            embed_suffix.append(' {' + f'"dialogue_history": "{history}", "current_turn": ')

        # 3) Tokenize the embedding suffix
        with nadd_eos(self.tokenizer), nadd_bos(self.tokenizer), left_padding(self.tokenizer):
            prompt_suffix_ids = self.tokenizer.batch_encode_plus(
                [e for e in embed_suffix],
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        # 4) Apply padding to the features
        batch = self.feature_extractor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            batch[self.feature_extractor.model_input_names[0]] = batch[
                self.feature_extractor.model_input_names[0]
            ].transpose(-2, -1)

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if prompt_prefix_ids is not None:
            batch["prompt_prefix_ids"] = prompt_prefix_ids['input_ids']
            batch["prompt_prefix_mask"] = prompt_prefix_ids['attention_mask']

        if prompt_suffix_ids is not None:
            batch["prompt_suffix_ids"] = prompt_suffix_ids['input_ids']
            batch["prompt_suffix_mask"] = prompt_suffix_ids['attention_mask']

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        return batch

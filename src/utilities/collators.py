from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
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

        labels = self.tokenizer.batch_encode_plus(
            [feature[self.text_path] for feature in features],
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

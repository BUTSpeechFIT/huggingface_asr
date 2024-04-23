import importlib
import json
from ctypes import c_bool
from dataclasses import field, make_dataclass
from functools import partial
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import DatasetDict
from transformers import (
    EarlyStoppingCallback,
    SequenceFeatureExtractor,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.utils import logging

from utilities.data_utils import audio_object_stripper
from utilities.general_utils import (
    FunctionReturnWrapper,
    resolve_attribute_from_nested_class,
)
from utilities.training_arguments import DataTrainingArguments, GeneralTrainingArguments

logger = logging.get_logger("transformers")


class GumbelTemperatureCallback(TrainerCallback):
    def __init__(self, gumbel_temperature_decay: float, min_gumbel_temperature: float, max_gumbel_temperature: float):
        super().__init__()
        self.gumbel_temperature_decay = gumbel_temperature_decay
        self.min_gumbel_temperature = min_gumbel_temperature
        self.max_gumbel_temperature = max_gumbel_temperature
        self.current_gumbel_temperature = max_gumbel_temperature

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.current_gumbel_temperature = self.max_gumbel_temperature
        kwargs["model"].set_gumbel_temperature(self.current_gumbel_temperature)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.current_gumbel_temperature = max(
            self.max_gumbel_temperature * self.gumbel_temperature_decay**state.global_step,
            self.min_gumbel_temperature,
        )
        kwargs["model"].set_gumbel_temperature(self.current_gumbel_temperature)


class DelayedStartWrapper:
    def __init__(self, callback: FunctionReturnWrapper, delay_steps: int):
        self.callback = callback
        self.start_at = delay_steps
        self.active = mp.Value(c_bool, False)

    def new_step(self, step: int):
        if step >= self.start_at and not self.active.value:
            logger.info(f"Activated preprocessing function: {str(self.callback.func)}")
            self.active.value = True

    def __call__(self, *args, **kwargs):
        if self.active.value:
            return self.callback(*args, **kwargs)
        return args[0]


class DataPreprocessingManagerCallback(TrainerCallback):
    def __init__(
        self,
        preprocessing_config: Dict[str, List[Dict]],
        dataset: DatasetDict,
        audio_column_name: str,
        feature_extractor: SequenceFeatureExtractor,
    ):
        super().__init__()
        self.dataset = dataset
        self.audio_column_name = audio_column_name
        self.transforms = {split: [] for split in preprocessing_config.keys()}
        for split, config_list in preprocessing_config.items():
            for config in config_list:
                if config["name"] == "feature_extractor":
                    fun = feature_extractor
                else:
                    module, attribute = config["name"].rsplit(".", 1)
                    fun = resolve_attribute_from_nested_class(importlib.import_module(module), attribute)(
                        **config["params"]
                    )

                self.transforms[split].append(
                    (
                        DelayedStartWrapper(
                            FunctionReturnWrapper(fun, config["return_behaviour"]), config["steps_before_activation"]
                        ),
                        config["fn_call_params"],
                    )
                )

    @staticmethod
    def transformer(audio: Union[np.ndarray, torch.Tensor], transforms: List[Tuple[DelayedStartWrapper, Dict]]):
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        for transform, fn_call_params in transforms:
            audio = transform(audio, **fn_call_params)
        return audio

    def default_transform(self, batch, transform_key):
        return {
            self.audio_column_name: [
                self.transformer(audio_object_stripper(audio), self.transforms[transform_key])
                for audio in batch[self.audio_column_name]
            ]
        }

    def propagate_state_to_transforms(self, state: TrainerState):
        for split_transforms in self.transforms.values():
            for transform in split_transforms:
                transform[0].new_step(state.global_step)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for split in self.dataset.keys():
            transform_key = "default_preprocessing" if split not in self.transforms else split
            self.dataset[split].set_transform(
                partial(self.default_transform, transform_key=transform_key),
                columns=[self.audio_column_name],
                output_all_columns=True,
            )
        self.propagate_state_to_transforms(state)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """This ensures that the preprocessing functions are aware of the correct step even when restarting."""
        self.propagate_state_to_transforms(state)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.propagate_state_to_transforms(state)


class AdditionalLossPrinterCallback(TrainerCallback):
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        state.__class__ = make_dataclass(
            "state_derived",
            [("additional_logs", List[List[float]], field(default_factory=list))],
            bases=(TrainerState,),
        )
        state.additional_logs = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if hasattr(state, "additional_logs") and len(state.additional_logs) > 0:
            enc_loss, dec_loss = torch.tensor(state.additional_logs).mean(axis=0)
            if state.is_local_process_zero:
                logs["enc_loss"] = float(enc_loss)
                logs["dec_loss"] = float(dec_loss)
            state.additional_logs = []


def init_callbacks(
    data_args: DataTrainingArguments,
    training_args: GeneralTrainingArguments,
    dataset: DatasetDict,
    feature_extractor: SequenceFeatureExtractor,
):
    callbacks = []
    if data_args.data_preprocessing_config:
        with open(data_args.data_preprocessing_config) as config_handle:
            callbacks.append(
                DataPreprocessingManagerCallback(
                    preprocessing_config=json.load(config_handle),
                    dataset=dataset,
                    audio_column_name=data_args.audio_column_name,
                    feature_extractor=feature_extractor,
                )
            )
    else:
        default_preprocessing = [
            {
                "name": "feature_extractor",
                "steps_before_activation": 0,
                "fn_call_params": {
                    "return_attention_mask": False,
                    "sampling_rate": 16000,
                    "return_tensors": "pt",
                },
                "return_behaviour": ["input_features[0]"],
            }
        ]
        callbacks.append(
            DataPreprocessingManagerCallback(
                preprocessing_config={"default_preprocessing": default_preprocessing},
                dataset=dataset,
                audio_column_name=data_args.audio_column_name,
                feature_extractor=feature_extractor,
            )
        )
    if training_args.early_stopping_patience > -1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    if training_args.track_ctc_loss:
        callbacks.append(AdditionalLossPrinterCallback())
    return callbacks

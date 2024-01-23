import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import tqdm
from datasets import DatasetDict
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    SpeechEncoderDecoderModel,
    Trainer,
)
from transformers.generation.utils import BeamSearchOutput
from transformers.utils import logging

from utilities.generation_utils import save_nbests, save_predictions
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
)

logger = logging.get_logger("transformers")


class FunctionReturnWrapper:
    def __init__(self, func: Callable, config: Dict):
        self.func = func
        self.return_config = config

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        if self.return_config is None:
            return result
        else:
            return self._process_return_config(self.return_config, result)

    @staticmethod
    def _process_return_config(return_config: Dict, result: Union[Dict, torch.Tensor]) -> Union[tuple[Any, ...], Any]:
        if isinstance(return_config, list):
            if all(isinstance(i, (int, str)) for i in return_config):
                output = tuple(  # nosec
                    eval(key, {}, result) if isinstance(key, str) else result[key] for key in return_config  # nosec
                )  # nosec
                if len(output) == 1:
                    return output[0]
                else:
                    return output
            else:
                raise ValueError("Invalid return configuration. Use a list of integers/strings.")
        else:
            raise ValueError("Invalid return configuration. Use None or a list of integers/strings.")


def resolve_attribute_from_nested_class(obj: Any, attr_spec: str) -> Any:
    for attr in attr_spec.split("."):
        try:
            obj = obj[attr]
        except (TypeError, KeyError):
            obj = getattr(obj, attr)
    return obj


def average_dicts(*dicts) -> Tuple[Dict, int]:
    result = {}

    # Count the number of dictionaries
    num_dicts = len(dicts)

    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value

    return result, num_dicts


def move_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {key: move_to_cpu(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(item) for item in obj)
    else:
        return obj


def postprocess_beam_outputs(outputs: BeamSearchOutput) -> Dict[str, Any]:
    for key in outputs:
        outputs[key] = move_to_cpu(outputs[key])
    outputs["joint_scores"] = outputs["scores"][::4]
    outputs["dec_scores"] = outputs["scores"][1::4]
    outputs["ctc_scores"] = outputs["scores"][2::4]
    outputs["external_lm_scores"] = outputs["scores"][3::4]
    outputs = dict(outputs)
    del outputs["scores"]
    del outputs["encoder_hidden_states"]
    del outputs["decoder_hidden_states"]
    return outputs


def do_evaluate(
    trainer: Union[Trainer, Seq2SeqTrainer],
    dataset: DatasetDict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    gen_args: Optional[GenerationArguments],
    training_args: GeneralTrainingArguments,
    data_args: DataTrainingArguments,
):
    if data_args.test_splits is None:
        return
    if isinstance(trainer, Seq2SeqTrainer) and isinstance(model, SpeechEncoderDecoderModel):
        trainer.args.per_device_eval_batch_size = math.ceil(
            trainer.args.per_device_eval_batch_size / gen_args.eval_beam_factor
        )
    for split in data_args.test_splits:
        if isinstance(trainer, Seq2SeqTrainer):
            predictions = trainer.predict(
                dataset[split],
                output_hidden_states=True,
                num_beams=model.generation_config.num_beams * gen_args.eval_beam_factor,
            )
        else:
            predictions = trainer.predict(
                dataset[split],
            )
        logger.info(f"Metrics for {split} split: {predictions.metrics}")
        save_predictions(
            tokenizer,
            predictions,
            f"{training_args.output_dir}/" f'predictions_{split}_wer{100 * predictions.metrics["test_wer"]:.2f}.csv',
        )


def do_generate(
    trainer: Seq2SeqTrainer,
    dataset: DatasetDict,
    model: SpeechEncoderDecoderModel,
    tokenizer: PreTrainedTokenizer,
    gen_args: GenerationArguments,
    data_args: DataTrainingArguments,
    gen_config: GenerationConfig,
):
    if data_args.test_splits is None:
        return

    gen_config.num_return_sequences = gen_args.num_predictions_to_return
    gen_config.return_dict_in_generate = True
    gen_config.num_beams = model.generation_config.num_beams * gen_args.eval_beam_factor
    gen_config.output_scores = True
    trainer.args.per_device_eval_batch_size = math.ceil(
        trainer.args.per_device_eval_batch_size / gen_args.eval_beam_factor
    )
    for split in data_args.test_splits:
        logger.info(f"Generating predictions for split: {split}")
        dataloader = trainer.get_eval_dataloader(dataset[split])
        n_bests = []
        scores = []
        labels = []
        outputs_agg = []
        for sample in tqdm.tqdm(dataloader):
            outputs = model.generate(generation_config=gen_config, **sample)
            if gen_args.save_output_states:
                outputs_agg.append(postprocess_beam_outputs(outputs))
            n_bests.append(outputs.sequences)
            scores.append(outputs.sequences_scores)
            labels.append(sample["labels"])
        save_nbests(
            gen_args.nbest_path_to_save + "_" + split,
            n_bests,
            scores,
            labels,
            tokenizer,
            group_size=gen_args.num_predictions_to_return,
            outputs=outputs_agg,
            batch_size=trainer.args.per_device_eval_batch_size,
        )

import json
import math
import os
import time
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    Wav2Vec2CTCTokenizer,
)
from transformers.generation.utils import BeamSearchOutput
from transformers.utils import logging

import utilities.data_utils as data_utils
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


def function_aggregator(fun_list):
    def wrapper(arg):
        for fun in reversed(fun_list):
            arg = fun(arg)
        return arg

    return wrapper


def text_transform_partial(f):
    def wrapped(*args2, **kwargs2):
        return f(*args2, **kwargs2, label_column="aux")["aux"]

    return wrapped


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
    if hasattr(gen_args, "override_for_evaluation") and gen_args.override_for_evaluation is not None:
        num_beams_orig = model.generation_config.num_beams
        model.generation_config.update_from_string(gen_args.override_for_evaluation)
        trainer.args.generation_num_beams = model.generation_config.num_beams
        if model.generation_config.num_beams != num_beams_orig:
            trainer.args.per_device_eval_batch_size = math.ceil(
                trainer.args.per_device_eval_batch_size / (model.generation_config.num_beams / num_beams_orig)
            )
    for split in data_args.test_splits:
        start_time = time.time()
        if isinstance(trainer, Seq2SeqTrainer):
            predictions = trainer.predict(
                dataset[split],
                output_hidden_states=True,
            )
        else:
            predictions = trainer.predict(
                dataset[split],
            )
        end_time = time.time()
        tokens_produced = (predictions.predictions != tokenizer.pad_token_id).sum()
        logger.info(f"Metrics for {split} split: {predictions.metrics}")
        logger.info(f"Time taken for evaluation: {end_time - start_time} seconds")
        logger.info(f"Tokens produced: {tokens_produced}")
        logger.info(f"Tokens per second: {tokens_produced / (end_time - start_time)}")

        if gen_args.post_process_predictions and data_args.text_transformations is not None:
            callable_transform = function_aggregator(
                [
                    text_transform_partial(
                        getattr(data_utils, transform_name, lambda x, label_column: {label_column: x})
                    )
                    for transform_name in data_args.text_transformations
                ]
            )
        else:
            callable_transform = None

        save_predictions(
            tokenizer,
            predictions,
            f"{training_args.output_dir}/" f'predictions_{split}_wer{100 * predictions.metrics["test_wer"]:.2f}.csv',
            callable_transform,
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


class CustomWav2Vec2CTCTokenizer(Wav2Vec2CTCTokenizer):  # nosec
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        replace_word_delimiter_char=" ",
        do_lower_case=False,
        target_lang=None,
        **kwargs,
    ):
        self._word_delimiter_token = word_delimiter_token

        self.do_lower_case = do_lower_case
        self.replace_word_delimiter_char = replace_word_delimiter_char
        self.target_lang = target_lang

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.vocab = json.load(vocab_handle)

        # if target lang is defined vocab must be a nested dict
        # with each target lang being one vocabulary
        if target_lang is not None:
            self.encoder = self.vocab[target_lang]
        else:
            self.encoder = self.vocab

        self.decoder = {v: k for k, v in self.encoder.items()}

        super(Wav2Vec2CTCTokenizer, self).__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            replace_word_delimiter_char=replace_word_delimiter_char,
            target_lang=target_lang,
            **kwargs,
        )

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
        **kwargs,
    ) -> List[str]:
        if "group_ctc_tokens" in kwargs:
            group_ctc_tokens = kwargs.pop("group_ctc_tokens")
        return super().batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            output_char_offsets=output_char_offsets,
            output_word_offsets=output_word_offsets,
            **kwargs,
        )


def prepare_tokenizer_for_ctc(tokenizer, default_sep_token=" "):  # nosec
    if isinstance(tokenizer, Wav2Vec2CTCTokenizer):
        return tokenizer

    if tokenizer.sep_token is None:
        logger.warning("This tokenizer does not have a separator token which is required for CTC training.")
        tokenizer.sep_token = default_sep_token
    vocab = tokenizer.get_vocab()

    # Replace every token with separator by space in vocabulary
    if tokenizer.sep_token != default_sep_token:
        vocab_changes = []
        for token in vocab:
            if tokenizer.sep_token in token:
                vocab_changes.append((token, token.replace(tokenizer.sep_token, " ")))
        for change in vocab_changes:
            vocab[change[1]] = vocab.pop(change[0])

    # Save vocab to tmp file
    with NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        # Write JSON object to the temporary file
        json.dump(vocab, tmp_file)

    wrapped_tokenizer = CustomWav2Vec2CTCTokenizer(
        tmp_file.name,
        unk_token=tokenizer.unk_token,
        pad_token=tokenizer.pad_token,
        word_delimiter_token=default_sep_token,
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token,
    )

    os.remove(tmp_file.name)

    return wrapped_tokenizer

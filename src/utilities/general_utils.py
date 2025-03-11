import math
from typing import Any, Callable, Dict, Optional, Tuple, Union
from functools import reduce
import json
from accelerate import Accelerator
import copy

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

import utilities.data_utils as data_utils
from utilities.generation_utils import save_nbests, save_predictions, save_predictions_json
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
)
from utilities.collators import WOZCollator

logger = logging.get_logger("transformers")

from typing import List, Iterator, Any, TypeVar

T = TypeVar('T')

def zip_uneven(*iterables: List[T]) -> Iterator[tuple[T, ...]]:
    """
    Zip through multiple iterables of uneven lengths, producing tuples that decrease
    in size as iterables are exhausted.
    
    Args:
        *iterables: Variable number of iterables to zip through
        
    Yields:
        Tuple containing elements from non-exhausted iterables only
        
    Example:
        >>> list(zip_uneven([1, 2, 3], [4, 5], [6]))
        [(1, 4, 6), (2, 5), (3,)]
    """
    # Create iterator for each input iterable
    iterators = [iter(it) for it in iterables]
    # Track which iterators are exhausted
    is_active = [True] * len(iterators)
    
    # Continue while any iterator is still active
    while any(is_active):
        result = []
        for i, iterator in enumerate(iterators):
            if is_active[i]:
                try:
                    value = next(iterator)
                    result.append(value)
                except StopIteration:
                    is_active[i] = False
        
        # Only yield if we found any values
        if result:
            yield tuple(result)


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
    if gen_args.override_for_evaluation is not None:
        num_beams_orig = model.generation_config.num_beams
        model.generation_config.update_from_string(gen_args.override_for_evaluation)
        trainer.args.generation_num_beams = model.generation_config.num_beams
        if model.generation_config.num_beams != num_beams_orig:
            trainer.args.per_device_eval_batch_size = math.ceil(
                trainer.args.per_device_eval_batch_size / (model.generation_config.num_beams / num_beams_orig)
            )
    for split in data_args.test_splits:
        if isinstance(trainer, Seq2SeqTrainer):
            predictions = trainer.predict(
                dataset[split],
                output_hidden_states=True,
            )
        else:
            predictions = trainer.predict(
                dataset[split],
            )
        logger.info(f"Metrics for {split} split: {predictions.metrics}")

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

        save_predictions_json(
            tokenizer,
            predictions,
            f"{training_args.output_dir}/" f'predictions_{split}.json',
        )

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

def do_generate_woz(
    trainer: Seq2SeqTrainer,
    dataset: DatasetDict,
    model: SpeechEncoderDecoderModel,
    tokenizer: PreTrainedTokenizer,
    gen_args: GenerationArguments,
    data_args: DataTrainingArguments,
    gen_config: GenerationConfig,
    collator: WOZCollator,
):
    if data_args.test_splits is None:
        return

    gen_config.return_dict_in_generate = True
    gen_config.num_beams = model.generation_config.num_beams * gen_args.eval_beam_factor
    trainer.args.per_device_eval_batch_size = math.ceil(
        trainer.args.per_device_eval_batch_size / gen_args.eval_beam_factor
    )
    model = model.cuda()
    model.eval()
    predictions = {}

    for split in data_args.test_splits:
        logger.info(f"Generating predictions for split: {split}")

        test_split = dataset[split]

        # this assumes that the dataset is completely sorted
        wav_id = None
        context_generated = []
        test = False
        for sample in tqdm.tqdm(test_split):
            # setup a new prediction bin
            if sample['wav_id'] != wav_id:
                #if test:
                #    break
                #else: test = True
                wav_id = sample['wav_id']
                predictions[wav_id] = []
                context_generated = []


            # save the original context 
            context_original = sample['context']['text']
            sample['context']['text'] = context_generated
            model_inputs = collator([sample])

            # prepare the labels
            labels = model_inputs['labels']
            labels[labels == -100] = tokenizer.pad_token_id
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            model_inputs = model_inputs.to(model.device)
            with torch.no_grad():
                outputs = model.generate(**model_inputs, generation_config=gen_config)['sequences'].cpu()
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            # now parse out the generated text 
            try:
                pred_dict = json.loads('{"transcript": ' + generated_text)
            except:
                print("Error parsing prediction")
                pred_dict = {'transcript': ''}

            predictions[wav_id].append({
                'context_gt': context_original,
                'text_gt': labels,
                'context_hyp': context_generated,
                'text_hyp': generated_text,
            })

            context_generated.append(pred_dict['transcript'])

        #tst = predictions[wav_id][-1]
        #print(tst['context_gt'])
        #print(tst['context_hyp'])
        #print(tst['text_gt'])
        #print(tst['text_hyp'])

DUMMY = "#dummy#"
class WavIDLoader(torch.utils.data.Dataset):
    def __init__(self, woz_split, accelerator, limit=None, batch_size_times_proc=None, split_name=None) -> None:
        self.data = woz_split
        self.conversations = {}

        # first get all the conversation ids
        logger.info("Preparing dataframe...")
        df = self.data.to_pandas().drop(columns=['audio'])
        logger.info(f"Preparing indices for the '{split_name if split_name else self.data}' split")
        self.index_dict = df.groupby('wav_id').apply(lambda x: x.index.tolist()).to_dict()
        self.ids = list(self.index_dict.keys())

        if limit is not None:
            self.ids = self.ids[:limit]

        # because distributed samplers are effin stupid, extend the ids to be divisible by the number of gpu*batch_size
        if batch_size_times_proc is not None and ((mod := len(self.ids) % batch_size_times_proc) != 0):
            append = [DUMMY + el for el in self.ids[-(batch_size_times_proc - mod):] ]
            self.ids.extend(append)

        # load and store each conversation in memory (the test sets are small)
        for wav_id in tqdm.tqdm(self.ids, desc="Loading conversations..", disable=not accelerator.is_local_main_process):
            idx = wav_id.split("#")[-1]
            rows = self.data.select(self.index_dict[idx]).sort('turn_index')
            self.conversations[wav_id] = list(rows)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.conversations[self.ids[idx]]


def do_generate_woz_batched(
    trainer: Seq2SeqTrainer,
    dataset: DatasetDict,
    model: SpeechEncoderDecoderModel,
    tokenizer: PreTrainedTokenizer,
    gen_args: GenerationArguments,
    data_args: DataTrainingArguments,
    training_args: GeneralTrainingArguments,
    gen_config: GenerationConfig,
    collator: WOZCollator,
    woz_use_gt_context: bool = False,
    constrained_beam_search: Optional[List[str]] = None,
):
    if data_args.test_splits is None:
        return

    if constrained_beam_search:
        force_words_ids = [ tokenizer([word], add_special_tokens=False).input_ids for word in constrained_beam_search ]

    gen_config.return_dict_in_generate = True
    gen_config.num_beams = model.generation_config.num_beams * gen_args.eval_beam_factor
    trainer.args.per_device_eval_batch_size = math.ceil(
        trainer.args.per_device_eval_batch_size / gen_args.eval_beam_factor
    )
    model.eval()
    accelerator = Accelerator(mixed_precision='bf16')
    model = accelerator.prepare(model)

    for split in data_args.test_splits:
        logger.info(f"Generating predictions for split: {split}")

        accelerator.wait_for_everyone()

        # first, create the dataset and dataloader
        test_split = WavIDLoader(
            dataset[split],
            accelerator,
            batch_size_times_proc=accelerator.num_processes*trainer.args.per_device_eval_batch_size,
            split_name=split,
        )

        data_loader = torch.utils.data.DataLoader(
                test_split,
                batch_size=trainer.args.per_device_eval_batch_size,
                collate_fn=lambda x: x,
                num_workers=0,
                drop_last=False,
        )

        data_loader, collator = accelerator.prepare(data_loader, collator)

        predictions = {}
        for base_batch in tqdm.tqdm(data_loader, position=0, disable=not accelerator.is_local_main_process):

            # sort the batch based on the length of the conversations so that we can freely use indices
            base_batch = sorted(base_batch, key=lambda x: len(x[1]), reverse=True)

            # get the ids in the batch and prepare the prediction dictionaries
            wav_ids_batch = [ w for w, _ in base_batch ]
            context_generated = {}

            for wav_id in wav_ids_batch:
                predictions[wav_id] = []
                context_generated[wav_id] = []

            batch_data = [ b for _, b in base_batch ]

            for batch in zip_uneven(*batch_data):

                # because the batches are uneven, we need to keep track of the current wav ids
                current_wav_ids = wav_ids_batch[:len(batch)]
                context_original = { wav_id: {} for wav_id in current_wav_ids }

                for sample, wav_id in zip(batch, current_wav_ids):
                    context_original[wav_id]['user'] = sample['context']['text'] # TODO FIX handle agent turns as well
                    context_original[wav_id]['agent'] = sample['context']['agent_text']
                    if not woz_use_gt_context:
                        sample['context']['text'] = context_generated[wav_id]

                model_inputs = collator(batch).to(model.device)

                # prepare the labels
                labels = model_inputs['labels']
                labels[labels == -100] = tokenizer.pad_token_id
                labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                with torch.no_grad():
                    if constrained_beam_search:
                        outputs = accelerator.unwrap_model(model).generate(**model_inputs, force_words_ids=force_words_ids, generation_config=gen_config)['sequences'].cpu()
                    else:
                        outputs = accelerator.unwrap_model(model).generate(**model_inputs, generation_config=gen_config)['sequences'].cpu()
                generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # now parse out the generated text 
                for i, (generated_text, wav_id) in enumerate(zip(generated_batch, current_wav_ids)):
                    try:
                        pred_dict = json.loads('{"transcript": ' + generated_text)
                    except:
                        logger.info("Error parsing prediction")
                        pred_dict = {'transcript': ''}

                    predictions[wav_id].append({
                        'turn_index': batch[i]['turn_index'],
                        'context_gt_user': context_original[wav_id]['user'],
                        'context_gt_agent': context_original[wav_id]['agent'],
                        'text_gt': labels[i],
                        'context_hyp_user': copy.deepcopy(context_generated[wav_id]),
                        'text_hyp': generated_text,
                    })

                    new_contex = pred_dict['transcript']
                    if type(new_contex) != str:
                        new_contex = ""
                    context_generated[wav_id].append(new_contex)

        predictions_all = accelerator.gather_for_metrics([predictions])
        if accelerator.is_main_process:

            # merge the predictions
            predictions_all = reduce(lambda x, y: {**x, **y}, predictions_all)
            predictions_all = { k: v for k, v in predictions_all.items() if DUMMY not in k }
            logger.info(f"Saving predictions for split: {split}")


            # save predictions to a json file
            with open(f"{training_args.output_dir}/" f'predictions_{split}_complete.json', 'w') as f:
                json.dump(predictions_all, f, indent=4)

            jga_json = {}
            json_errors = 0
            for key, preds in predictions_all.items():
                jga_json[key] = []

                for pred in preds:
                    text_hyp = pred['text_hyp']
                    turn_dict = {'response': "", 'state': {}, 'active_domains': []}
                    json_error = False
                    try:
                        pred_dict = json.loads('{"transcript": ' + text_hyp)

                        domains = pred_dict['domains']
                        if type(domains) == list:
                            for elem in domains:
                                if type(elem) != str:
                                    domains = []
                                    json_error = True
                                    logger.info("Error parsing domains")
                                    break
                        else:
                            domains = []
                            json_error = True
                            logger.info("Error parsing domains")

                        slots = pred_dict['slots']
                        if type(slots) == dict:
                            for _, v in slots.items():
                                if type(v) != dict:
                                    slots = {}
                                    json_error = True
                                    logger.info("Error parsing slots")
                                    break
                                else:
                                    for _, vv in v.items():
                                        if type(vv) != str:
                                            slots = {}
                                            json_error = True
                                            logger.info("Error parsing slots")
                                            break # TODO Goto
                        else:
                            slots = {}
                            json_error = True
                            logger.info("Error parsing slots")

                        turn_dict['active_domains'] = domains
                        turn_dict['state'] = slots

                    except:
                        json_error = True
                        logger.info("Error parsing prediction")

                    if json_error: json_errors += 1

                    jga_json[key].append(turn_dict)

            # save the file
            with open(f"{training_args.output_dir}/" f'predictions_{json_errors}err_{split}_states.json', 'w') as f:
                json.dump(jga_json, f, indent=4)

        # delete the objects to avoid memory errors
        accelerator.wait_for_everyone()
        del predictions
        if accelerator.is_main_process:
            del predictions_all
        del test_split
        del data_loader

import os
import subprocess  # nosec
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BatchFeature, Seq2SeqTrainer, Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import _is_peft_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.utils import (
    is_datasets_available,
    is_torch_tpu_available,
    is_torch_xla_available,
    logging,
)
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor

from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoder,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from utilities.callbacks import GumbelTemperatureCallback
from utilities.eval_utils import extract_err_rate_from_sclite
if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    # pylint: disable=import-error
    import torch_xla.core.xla_model as xm

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class AdditionalLossTrackerTrainer(Seq2SeqTrainer):
    """Custom trainer to log both losses"""

    def compute_loss(
            self, model: JointCTCAttentionEncoderDecoder, inputs: BatchFeature, return_outputs=False
    ) -> Union[float, Tuple[float, BatchFeature]]:
        """
        MAX: Subclassed to compute training accuracy.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if hasattr(self.state, "additional_logs"):
            self.state.additional_logs.append([outputs.enc_loss.mean(), outputs.dec_loss.mean()])

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

class RNNTTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        labels = inputs.pop("labels")
        out = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        return (None, out[1], labels)

class GradAwareTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_norm_thr = 200

    def get_grad_norm(self, model: nn.Module) -> torch.Tensor:
        total_norm = 0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # pylint: disable=no-member
        loss = super().training_step(model, inputs)

        if loss == 0:
            self.optimizer.zero_grad(set_to_none=True)

        total_norm = self.get_grad_norm(model)
        if total_norm > self.grad_norm_thr:
            logger.warning(f"Gradient norm: {total_norm}, loss: {loss.item()}")

            self.optimizer.zero_grad(set_to_none=True)
            loss -= loss

        if torch.isnan(torch.tensor(total_norm)):
            logger.warning("Gradient norm is NaN")
            if not os.path.exists("nan_optimizer.pkl"):
                torch.save(self.optimizer.state_dict(), "nan_optimizer.pkl")
                torch.save(inputs, "nan_inputs.pkl")
                torch.save(model, "nan_model.pkl")
            self.optimizer.zero_grad(set_to_none=True)
            loss -= loss
        return loss


class CustomSeq2SeqTrainer(GradAwareTrainer, Seq2SeqTrainer):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dev_output_dir = os.path.join(self.args.output_dir, "dev")
        os.makedirs(self.dev_output_dir, exist_ok=True)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # pylint: disable=no-member
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        if self.args.use_sclite_for_metrics and self.is_in_train and self.accelerator.is_main_process:
            pred_str = self.tokenizer.batch_decode(output.predictions, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(output.label_ids, skip_special_tokens=True)

            output_dir = os.path.join(self.dev_output_dir, str(self.state.global_step), metric_key_prefix)
            os.makedirs(output_dir, exist_ok=True)

            sclite_files = [f"{output_dir}/{type}.trn" for type in ["hyp", "ref"]]
            for strings, file_to_save in zip([pred_str, label_str], sclite_files):
                with open(file_to_save, "w") as file_handler:
                    for index, string in enumerate(strings):
                        file_handler.write(f"{string} (utterance_{index})\n")

            sclite_cmd = f"sclite -F -D -i wsj -r {sclite_files[1]} trn -h {sclite_files[0]} trn -o snt sum dtl"
            process = subprocess.Popen(sclite_cmd.split())  # nosec
            try:
                process.wait(60)
                err = extract_err_rate_from_sclite(open(f'{sclite_files[0]}.sys').read())
                output.metrics['eval_sclite_wer'] = err
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("Sclite evaluation timed out.")
            except ValueError:
                logger.warning("Could extract wer from sclite output.")
        return output


class SaveFeatureExtractorTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        self.data_collator.feature_extractor.save_pretrained(output_dir)


class SSLTrainer(GradAwareTrainer, SaveFeatureExtractorTrainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.gumbel_callback = None
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, GumbelTemperatureCallback):
                self.gumbel_callback = callback

        self.can_return_loss = True
        self.metadata = {"train": {}, "eval": {}}
        self.feature_extractor = feature_extractor

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        additional_stats, num_losses = self.gather_additional_statistics(inputs, outputs)
        stats_object = "train" if self.model.training else "eval"
        for key in additional_stats.keys():
            if key not in self.metadata[stats_object].keys():
                self.metadata[stats_object][key] = 0
            self.metadata[stats_object][key] += additional_stats[key]
        loss /= num_losses.sum()

        return (loss, outputs) if return_outputs else loss

    def gather_additional_statistics(self, inputs, outputs):
        additional_logs = {}
        num_losses = inputs["mask_time_indices"].sum(dim=1)
        sub_attention_mask = inputs.pop("attention_mask", None)
        sub_attention_mask = (
            sub_attention_mask
            if sub_attention_mask is not None
            else torch.ones_like(inputs["mask_time_indices"], device=inputs["mask_time_indices"].device)
        )

        input_lens = self.model._get_feat_extract_output_lengths(sub_attention_mask.sum(dim=1))

        percent_masked = (num_losses / input_lens).mean()

        if outputs.contrastive_loss:
            additional_logs["contrastive_loss"] = outputs.contrastive_loss
            additional_logs["diversity_loss"] = outputs.diversity_loss
            additional_logs["avg_ppl"] = outputs.codevector_perplexity
            additional_logs["gumbel_temperature"] = torch.tensor(
                self.gumbel_callback.current_gumbel_temperature, device=inputs["mask_time_indices"].device
            )
        if outputs.codevector_perplexity is not None:
            additional_logs["%_codebook_used"] = outputs.codevector_perplexity
        if outputs.diversity_loss is not None:
            additional_logs["%_unique_labels"] = outputs.diversity_loss
        additional_logs["%_mask_idx"] = percent_masked
        additional_logs["num_losses"] = num_losses.sum(dtype=torch.float32)

        for key in additional_logs.keys():
            additional_logs[key] = additional_logs[key].detach()

        return additional_logs, num_losses

    @staticmethod
    def normalize_additional_logs(additional_logs, normalizer):
        for key in additional_logs.keys():
            if key != "num_losses":
                if "loss" in key and "num_losses" in additional_logs.keys():
                    additional_logs[key] = additional_logs[key] / additional_logs["num_losses"]
                else:
                    additional_logs[key] = round(
                        additional_logs[key] / normalizer,
                        4,
                    )
        if "num_losses" in additional_logs.keys():
            del additional_logs["num_losses"]
        return additional_logs

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self._eval_dataloader
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = self.accelerator.prepare(eval_dataloader)
            return self._eval_dataloader

        return self.accelerator.prepare(eval_dataloader)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            for metric in self.metadata["train"].keys():
                metric_value = self._nested_gather(self.metadata["train"][metric]).mean().item()
                logs[metric] = round(metric_value / (self.state.global_step - self._globalstep_last_logged), 4)
                self.metadata["train"][metric] -= self.metadata["train"][metric]

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def evaluation_loop(
            self,
            eval_dataloader: DataLoader,
            *args,
            **kwargs,
    ) -> EvalLoopOutput:
        # pylint: disable=no-member
        output = super().evaluation_loop(eval_dataloader, *args, **kwargs)
        for metric in self.metadata["eval"].keys():
            metric_value = self._nested_gather(self.metadata["eval"][metric]).sum().item()
            output.metrics[metric] = round(
                metric_value / (output.num_samples / self.args.eval_batch_size * max(1, self.args.n_gpu)), 4
            )
            self.metadata["eval"][metric] -= self.metadata["eval"][metric]
        return output


class WhisperLongFormTrainer(CustomSeq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
                "labels" in generation_inputs
                and "decoder_input_ids" in generation_inputs
                and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                if inputs["input_features"].shape[-1] != model.config.max_source_positions * model.get_encoder().conv1.stride[0] * model.get_encoder().conv2.stride[0]:
                    logger.warning("Skipping loss calculation for long form segment!")
                    loss = None
                else:
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if self.label_smoother is not None:
                        loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

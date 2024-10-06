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

from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoder,
)
from utilities.callbacks import GumbelTemperatureCallback

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


class GradAwareTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_norm_thr = 100

    def get_grad_norm(self, model: nn.Module) -> torch.Tensor:
        total_norm = 0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
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

        if self.args.use_sclite_for_metrics and self.is_in_train:
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
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("Sclite evaluation timed out.")

        return output


class SSLTrainer(GradAwareTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
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

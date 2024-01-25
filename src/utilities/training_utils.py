from typing import Callable, List

from datasets import Dataset
from packaging import version
from torch import Tensor, nn
from transformers import (
    BatchFeature,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollator
from transformers.dependency_versions_check import dep_version_check

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    is_fairscale_available,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_callback import TrainerCallback
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoder,
)
from utilities.callbacks import GumbelTemperatureCallback

# isort: on


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    # pylint: disable=import-error
    from apex import amp

if is_torch_tpu_available(check_device=False):
    # pylint: disable=import-error
    import torch_xla.core.xla_model as xm

if is_fairscale_available():
    dep_version_check("fairscale")

if is_sagemaker_mp_enabled():
    # pylint: disable=import-error
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
    from transformers.trainer_pt_utils import smp_forward_backward

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import DistributedType

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch._C as _C
from torch.overrides import get_default_nowrap_functions

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def _convert(ret, cls):
    if cls is Tensor:
        return ret

    if isinstance(ret, Tensor) and not isinstance(ret, cls):
        ret = ret.as_subclass(cls)

    if isinstance(ret, (tuple, list)):
        # Also handles things like namedtuples
        ret = type(ret)(_convert(r, cls) for r in ret)

    return ret


class MetadataTensor(torch.Tensor):
    def __new__(cls, input_tensor, metadata=None):
        obj = super(MetadataTensor, cls).__new__(cls, input_tensor)
        obj.metadata = metadata if metadata else {}
        return obj

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying tensor
        return getattr(self.data, attr)

    def __add__(self, other):
        result = super(MetadataTensor, self).__add__(other)

        # Update metadata based on the addition operation
        if isinstance(other, MetadataTensor):
            result.metadata = {**self.metadata, **other.metadata}
        else:
            result.metadata = self.metadata.copy()

        return MetadataTensor(result, result.metadata)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.

        One corollary to this is that you need coverage for torch.Tensor
        methods if implementing __torch_function__ for subclasses.

        We recommend always calling ``super().__torch_function__`` as the base
        case when doing the above.

        While not mandatory, we recommend making `__torch_function__` a classmethod.
        """
        if kwargs is None:
            kwargs = {}
        if len(args) == 2 and not (isinstance(args[0], MetadataTensor)):
            args = (MetadataTensor(args[0], args[1].metadata), args[1])
            args[1].metadata = None
        elif len(args) == 2 and (isinstance(args[0], MetadataTensor)) and (isinstance(args[0], MetadataTensor)):
            m1 = args[0].metadata
            m2 = args[1].metadata
            for k in m1.keys():
                func(m1[k], m2[k])

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        with _C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return _convert(ret, cls)


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


class SSLTrainer(Trainer):
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
        self.gumbel_callback = [
            callback for callback in self.callback_handler.callbacks if isinstance(callback, GumbelTemperatureCallback)
        ][0]

    @staticmethod
    def multiply_grads(params, c):
        """Multiplies grads by a constant *c*."""
        for p in params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    @staticmethod
    def get_grad_norm(params, scale=1):
        """Compute grad norm given a gradient scale."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = (p.grad.detach().data / scale).norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[torch.Tensor, Wav2Vec2ForPreTrainingOutput]:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        additional_logs = {}
        num_losses = inputs["mask_time_indices"].sum()
        sub_attention_mask = inputs.pop("sub_attention_mask", None)
        sub_attention_mask = (
            sub_attention_mask if sub_attention_mask is not None else torch.ones_like(inputs["mask_time_indices"])
        )
        percent_masked = num_losses / sub_attention_mask.sum()
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss / num_losses
        additional_logs["constrast_loss"] = outputs.contrastive_loss / num_losses
        additional_logs["div_loss"] = outputs.diversity_loss / num_losses
        additional_logs["%_mask_idx"] = percent_masked / self.accelerator.num_processes
        additional_logs["ppl"] = outputs.codevector_perplexity
        additional_logs["temp"] = torch.tensor(self.gumbel_callback.current_gumbel_temperature)

        for key in additional_logs.keys():
            additional_logs[key] = additional_logs[key].detach()
        # pylint: disable=no-member
        if self.accelerator.state.num_processes > 1:
            num_losses = self.accelerator.gather_for_metrics(num_losses).sum()
            # pylint: disable=no-member
            gradient_multiplier = self.accelerator.state.num_processes / num_losses
            self.multiply_grads(model.module.parameters(), gradient_multiplier)
        else:
            self.multiply_grads(model.parameters(), 1 / num_losses)

        loss.additional_logs = additional_logs
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        _grad_norm = self.accelerator.clip_grad_norm_(
            model.parameters(),
            self.args.max_grad_norm,
        )

        if is_accelerate_available() and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            grad_norm = model.get_global_grad_norm()
        else:
            grad_norm = _grad_norm.item() if _grad_norm is not None else None
        additional_logs["gradient_norm"] = torch.tensor(grad_norm)

        loss_detached = loss.detach() / self.args.gradient_accumulation_steps
        loss_detached_with_metadata = MetadataTensor(loss_detached, metadata=additional_logs)
        return loss_detached_with_metadata

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            for item in tr_loss.metadata.keys():
                item_scalar = self._nested_gather(tr_loss.metadata[item]).mean().item()
                tr_loss.metadata[item] -= tr_loss.metadata[item]
                logs[item] = round(item_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
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

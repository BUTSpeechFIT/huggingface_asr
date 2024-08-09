"""Main training script for training of CTC ASR models."""
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)
from transformers.utils import logging
from whisper_ctc import CustomWhisperEncoderLayer, WhisperEncoderForCTC

from utilities.callbacks import init_callbacks
from utilities.collators import SpeechCollatorWithPadding
from utilities.data_utils import get_dataset
from utilities.eval_utils import compute_metrics_ctc, ctc_beam_decode, ctc_greedy_decode
from utilities.general_utils import do_evaluate
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
)


@dataclass
class CustomModelArguments(ModelArguments):
    llm_model: Optional[str] = field(default="google/gemma-2b-it", metadata={"help": "The model to use for the LLM."})


if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser(
        (CustomModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments)
    )

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Collect, preprocess dataset and extract evaluation dataset
    dataset, training_eval_dataset = get_dataset(
        data_args=data_args,
        len_column=training_args.length_column_name,
    )

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model)
    tokenizer.padding_side = "right"

    # 3. Instantiate model
    model = WhisperEncoderForCTC.from_pretrained(model_args.from_pretrained)
    config = model.config

    llm = AutoModelForCausalLM.from_pretrained(model_args.llm_model)

    config.out_dim = llm.config.hidden_size
    model.additional_layer = CustomWhisperEncoderLayer(config)
    model.lm_head = llm.lm_head
    model.lm_head.weight.requires_grad = False
    model.config.additional_layer = True
    model.config.ctc_zero_infinity = True
    model.config.blank_token_id = tokenizer.pad_token_id
    for module in [
        *model.encoder.layers[: int(len(model.encoder.layers) * 2 / 3)],
        model.encoder.conv1,
        model.encoder.conv2,
    ]:
        for param in module.parameters():
            param.requires_grad = False
    del llm
    # 4. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)

    # 5. Initialize data collator
    data_collator = SpeechCollatorWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        model_input_name=model.main_input_name,
        mask_unks=training_args.mask_unks,
        pad_to_multiple_of=data_args.pad_to_multiples_of,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=training_eval_dataset,
        data_collator=data_collator,
        preprocess_logits_for_metrics=(
            lambda predictions, labels: ctc_beam_decode(
                predictions, labels, tokenizer, training_args.generation_num_beams
            )
        )
        if training_args.generation_num_beams is not None and training_args.generation_num_beams > 1
        else ctc_greedy_decode,
        compute_metrics=lambda pred: compute_metrics_ctc(tokenizer, pred, gen_args.wandb_predictions_to_save),
    )

    # 6. Train
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    # 7. Evaluation
    if training_args.do_evaluate:
        do_evaluate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            data_args=data_args,
            training_args=training_args,
        )

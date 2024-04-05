"""Main training script for training of attention based encoder decoder ASR models."""
import sys

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    HfArgumentParser,
    MarianConfig,
    MarianMTModel,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

from transformers.utils import logging

from utilities.callbacks import init_callbacks
from utilities.collators import SpeechCollatorWithPadding
from utilities.data_utils import get_dataset
from utilities.eval_utils import compute_metrics_translation
from utilities.model_utils import average_checkpoints_torch
from utilities.general_utils import do_evaluate, do_generate
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
)


if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 0. prepare the how2 dataset object..
    dataset, training_eval_dataset = get_dataset(
        datasets_creation_config_path=data_args.datasets_creation_config,
        dataset_name=data_args.dataset_name,
        dataset_config=data_args.dataset_config,
        data_dir=data_args.data_dir,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        writer_batch_size=data_args.writer_batch_size,
        sampling_rate=data_args.sampling_rate,
        max_input_len=data_args.max_duration_in_seconds,
        min_input_len=data_args.min_duration_in_seconds,
        len_column=training_args.length_column_name,
        text_column=data_args.text_column_name,
        audio_column=data_args.audio_column_name,
        train_split=data_args.train_split,
        validation_split=data_args.validation_split,
        text_transformations=data_args.text_transformations,
        split_long_segments_to_chunks=data_args.split_long_segments_to_chunks,
        validation_slice_str=data_args.validation_slice,
        cut_validation_from_train=data_args.cut_validation_from_train,
        seed=data_args.validation_slice_seed,
        reshuffle_at_start=data_args.reshuffle_at_start,
        skip_audio_processing=True,
    )

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor and tokenizer
    #tokenizer_source = AutoTokenizer.from_pretrained(training_args.tokenizer_name)
    #tokenizer_source = AutoTokenizer.from_pretrained('pirxus/how2_en_unigram8000_tc')
    #tokenizer_target = AutoTokenizer.from_pretrained('pirxus/how2_pt_unigram8000_tc')
    tokenizer_source = AutoTokenizer.from_pretrained('pirxus/how2_en_bpe8000_tc')
    tokenizer_target = AutoTokenizer.from_pretrained('pirxus/how2_pt_bpe8000_tc')


    def preprocess_function(examples):
        inputs = examples['transcription']
        targets = examples['translation']

        tokenized_inputs = tokenizer_source(inputs)
        labels = tokenizer_target(targets)['input_ids']

        model_inputs = {
                'input_ids': tokenized_inputs['input_ids'],
                'attention_mask': tokenized_inputs['attention_mask'],
                'labels': labels,
                }

        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    logger.info(f"Finished tokenizing dataset: {tokenized_dataset}")

    # 3. Instantiate model
    base_model_config = {
        #"encoder_layerdrop": 0.0,
        "pad_token_id": tokenizer_source.pad_token_id,
        "encoder_pad_token_id": tokenizer_source.pad_token_id,
        "decoder_vocab_size": len(tokenizer_source),
        #"vocab_size": len(tokenizer_source), # s2t specific
        "lsm_factor": model_args.lsm_factor,
        "shared_lm_head": model_args.shared_lm_head,
        "decoder_start_token_id": tokenizer_target.bos_token_id,
        "decoder_pos_emb_fixed": model_args.decoder_pos_emb_fixed,
        "eos_token_id": tokenizer_target.eos_token_id,
        "pad_token_id": tokenizer_target.pad_token_id,
    }

    config = MarianConfig(
            vocab_size=tokenizer_source.vocab_size,
            d_model=256,
            encoder_layers=6,
            decoder_layers=6,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            decoder_ffn_dim=2048,
            encoder_ffn_dim=2048,
            activation_function='gelu_new',
            attention_dropout=0.1,
            activation_dropout=0.1,
            scale_embedding=True,
            forced_eos_token_id=tokenizer_target.eos_token_id,
            share_encoder_decoder_embeddings=False,
            eos_token_id=tokenizer_target.eos_token_id,
            pad_token_id=tokenizer_target.pad_token_id,
            decoder_start_token_id=tokenizer_target.bos_token_id,
            )
    #config = config.update(base_model_config)

    if model_args.from_pretrained:
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints_torch(model_path)

        model = MarianMTModel.from_pretrained(model_path)
    else:
        model = MarianMTModel(config=config)

    logger.info(f"Finished loading model {model}")

    # 4. Update generation config
    gen_config = GenerationConfig(
        bos_token_id=tokenizer_target.bos_token_id,
        pad_token_id=tokenizer_source.pad_token_id,
        decoder_start_token_id=tokenizer_target.bos_token_id,
        decoder_end_token_id=tokenizer_target.eos_token_id,
        length_penalty=gen_args.length_penalty,
        early_stopping=gen_args.early_stopping,
        eos_token_id=tokenizer_target.eos_token_id,
        max_length=gen_args.max_length,
        num_beams=gen_args.num_beams,
    )

    logger.info(f"Model updating generation config:\n {str(gen_config)}")
    training_args.generation_max_length = gen_args.max_length
    training_args.generation_num_beams = gen_args.num_beams
    model.generation_config = gen_config


    # 5. Initialize callbacks
    #callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)
    callbacks = []
    if training_args.early_stopping_patience > -1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))

    # 6. Initialize data collator
    #data_collator = SpeechCollatorWithPadding(
    #    feature_extractor=feature_extractor,
    #    tokenizer=tokenizer,
    #    padding=True,
    #    sampling_rate=data_args.sampling_rate,
    #    audio_path=data_args.audio_column_name,
    #    text_path=data_args.text_column_name,
    #    model_input_name=model.main_input_name,
    #)

    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer_source,
            model=model,
            )

    # 7. Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        #callbacks=callbacks,
        train_dataset=tokenized_dataset[data_args.train_split],
        eval_dataset=tokenized_dataset[data_args.validation_split],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics_translation(tokenizer_target, pred, gen_args.wandb_predictions_to_save),
    )

    # 8. Train model
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    # 9. Evaluation
    if training_args.do_evaluate:
        do_evaluate(
            trainer=trainer,
            dataset=tokenized_dataset,
            model=model,
            tokenizer=tokenizer_target,
            gen_args=gen_args,
            training_args=training_args,
            data_args=data_args,
        )
    # 10. N-best generation
    #if training_args.do_generate:
    #    do_generate(
    #        trainer=trainer,
    #        dataset=tokenized_dataset,
    #        model=model,
    #        tokenizer=tokenizer_target,
    #        gen_args=gen_args,
    #        data_args=data_args,
    #        eos_token_id=tokenizer_target.eos_token_id,
    #        gen_config=gen_config,
    #    )

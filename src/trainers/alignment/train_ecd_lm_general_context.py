"""Main training script for the encoder -> connector -> decoder-only LM architecture """
import sys


from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    Seq2SeqTrainer,
    Blip2QFormerConfig,
    WhisperForConditionalGeneration,
)
from transformers.utils import logging
import torch

from utilities.callbacks import init_callbacks
from utilities.collators import GeneralContextCollator
from utilities.data_utils import get_dataset
from utilities.eval_utils import compute_metrics_fisher_turns
from utilities.model_utils import average_checkpoints as average_checkpoints
from utilities.general_utils import do_evaluate, do_generate
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
    ConnectorArguments
)

from models.old_alignment import AlignmentConfig
from models.aligned_decoder_lm import SpeechEncoderConnectorLMDecoder
from trainers.alignment.train_ecd_lm import WavLMModelWrapper

from peft import LoraConfig, get_peft_model, replace_lora_weights_loftq


if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments, ConnectorArguments))

    model_args, data_args, training_args, gen_args, conn_args = parser.parse_args_into_dataclasses()

    # 0. prepare the how2 dataset object..
    # 1. Collect, preprocess dataset and extract evaluation dataset
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
        flatten_fisher=True,
    )

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.tokenizer_name,
        add_eos_token = True,
    )
    if not hasattr(tokenizer, 'pad_token_id'): # FIXME
        tokenizer.pad_token_id = tokenizer(tokenizer.pad_token)['input_ids'][0]

    # 3. Instantiate model
    # -- load the asr encoder
    if 'whisper' in model_args.base_encoder_model:
        encoder = WhisperForConditionalGeneration.from_pretrained(
            model_args.base_encoder_model,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        )
        d_model = encoder.config.d_model
        sample_rate = 100
    elif 'wavlm' in model_args.base_encoder_model:
        encoder = WavLMModelWrapper.from_pretrained(
            model_args.base_encoder_model,
            torch_dtype=torch.bfloat16,
        )
        encoder.config.apply_spec_augment = False
        encoder.config.layer_to_extract = model_args.layer_to_extract
        d_model = encoder.config.hidden_size
        sample_rate = 16000
    else:
        raise NotImplementedError('only Whisper and WavLm are supported')

    decoder = AutoModelForCausalLM.from_pretrained(
        model_args.base_decoder_model,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )

    # set up lora for the decoder
    if conn_args.decoder_lora:
        lora_config = LoraConfig(task_type='CAUSAL_LM', target_modules='all-linear')
        decoder = get_peft_model(decoder, lora_config)
        replace_lora_weights_loftq(decoder)

    # -- prepare the connector
    if model_args.from_config:
        apmo_config = AlignmentConfig.from_pretrained(model_args.from_config)
    else:

        qformer_config = Blip2QFormerConfig(
                hidden_size=conn_args.conn_hidden_size,
                num_hidden_layers=conn_args.conn_layers,
                num_attention_heads=conn_args.conn_attn_heads,
                intermediate_size=conn_args.qf_intermediate_size,
                hidden_act='gelu_new',
                cross_attention_frequency=1,
                encoder_hidden_size=d_model
            )

        apmo_config = AlignmentConfig(
                encoder_config=encoder.config,
                qformer_config=qformer_config,
                lm_config=decoder.config,
                num_query_tokens=conn_args.n_queries,
                mm_pooling=conn_args.qf_mm_pooling,
                mm_loss_weight=conn_args.qf_mm_loss_weight,
                connector_type=conn_args.connector_type,
                downsampling_factor=conn_args.downsampling_factor,
                prompt_tuning_prefix_len=conn_args.prompt_tuning_prefix_len,
                prompt_tuning_suffix_len=conn_args.prompt_tuning_suffix_len,
                init_prompt_from_embeds=conn_args.init_prompt_from_embeds,
                prompt_tuning_prefix_init=conn_args.prompt_tuning_prefix_init,
                prompt_tuning_suffix_init=conn_args.prompt_tuning_suffix_init,
                freeze_encoder=model_args.freeze_encoder,
            )

    # get the initialization point for the soft prompts if specified so
    # TODO: check the soft prompt implementation

    if model_args.from_pretrained:
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints(model_path)

        config = AlignmentConfig.from_pretrained(model_path)
        logger.info(f"Loading model from pretrained checkpoint...")
        
        model = SpeechEncoderConnectorLMDecoder.from_pretrained(model_path, config, encoder, decoder, tokenizer)

        if model_args.freeze_encoder:
            model.freeze_encoder()
        else:
            model.unfreeze_encoder()

    else:
        model = SpeechEncoderConnectorLMDecoder(encoder=encoder, decoder=decoder, config=apmo_config, freeze_decoder= not conn_args.decoder_lora, tokenizer=tokenizer)

    logger.info(f"Finished loading model {model}")

    # 4. Update generation config
    bos = decoder.config.decoder_start_token_id if tokenizer.bos_token_id is None else tokenizer.bos_token_id
    gen_config = GenerationConfig(
        bos_token_id=bos,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=bos,
        decoder_end_token_id=tokenizer.eos_token_id,
        length_penalty=gen_args.length_penalty,
        early_stopping=gen_args.early_stopping,
        eos_token_id=tokenizer.eos_token_id,
        #max_length=gen_args.max_length if gen_args.max_new_tokens is None else None,
        num_beams=gen_args.num_beams,
        max_new_tokens=gen_args.max_new_tokens,
    )

    logger.info(f"Model updating generation config:\n {str(gen_config)}")
    #training_args.generation_max_length = gen_args.max_length
    training_args.generation_num_beams = gen_args.num_beams
    model.generation_config = gen_config
    model.decoder.generation_config = gen_config
    if hasattr(model.decoder, 'decoder'):
        model.decoder.decoder.generation_config = gen_config


    # 5. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)

    # 6. Initialize data collator
    data_collator = GeneralContextCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        model_input_name=model.main_input_name,
        context_prefix=data_args.fisher_context_prefix,
        prompt_prefix=conn_args.prompt_prefix,
        prompt_suffix=conn_args.prompt_suffix,
        max_context=data_args.fisher_max_context,
        context_trunc_to_shortest=data_args.fisher_context_trunc_to_shortest,
        pad_to_multiple_of=int(sample_rate/2),
    )

    if gen_args.no_metrics:
        # bypasses decoding in the eval loop, speeding up the evaluation significantly. We only
        # get the eval loss this way as a metric
        c_metrics = None
    else:
        c_metrics = lambda pred: compute_metrics_fisher_turns(tokenizer, pred, gen_args.wandb_predictions_to_save, remove_spk_tags=True) 

    # 7. Initialize trainer
    trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            callbacks=callbacks,
            train_dataset=dataset[data_args.train_split],
            eval_dataset=training_eval_dataset,
            data_collator=data_collator,
            compute_metrics=c_metrics,
    )

    # 8. Train model
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    # 9. Evaluation
    if training_args.do_evaluate:
        do_evaluate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            training_args=training_args,
            data_args=data_args,
        )
    # 10. N-best generation
    if training_args.do_generate:
        do_generate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            data_args=data_args,
            gen_config=gen_config,
        )

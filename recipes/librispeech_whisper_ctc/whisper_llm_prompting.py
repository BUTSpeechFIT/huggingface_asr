"""Main training script for training of CTC ASR models."""
import copy
import sys

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
from local_utils import (
    CustomCollator,
    CustomModelArgumentsPrompting,
    compute_metrics_ctc,
    ctc_greedy_decode,
    do_evaluate,
    get_token_subset,
)
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
)
from transformers.utils import logging
from whisper_ctc import WhisperEncoderForCTC

from utilities.callbacks import init_callbacks
from utilities.data_utils import get_dataset
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
)


class LearnableBlankLinear(torch.nn.Module):
    def __init__(self, frozen_linear, blank_id):
        super().__init__()
        self.frozen_linear = frozen_linear
        self.blank_id = blank_id
        self.blank_projection = torch.nn.Linear(frozen_linear.in_features, 1)
        self.frozen_linear.weight.requires_grad = False

    def forward(self, x):
        out = self.frozen_linear(x)
        out[..., self.blank_id] = self.blank_projection(x).squeeze(dim=-1)
        return out


def get_model(m_args: CustomModelArgumentsPrompting):
    llm = AutoModelForCausalLM.from_pretrained(m_args.llm_model)
    new_model = WhisperEncoderForCTC.from_pretrained(
        m_args.from_pretrained, llm_dim=llm.config.hidden_size, sub_sample=True
    )
    llm_head = copy.deepcopy(llm.lm_head)
    unwanted_tokens_mask = np.ones((llm_head.weight.shape[0],), dtype=bool)
    unwanted_tokens_mask[removed_token_ids] = False
    new_model.config.blank_token_id = tokenizer.pad_token_id
    new_model.config.bos_token_id = tokenizer.bos_token_id
    new_model.config.eos_token_id = tokenizer.eos_token_id
    new_model.config.pad_token_id = tokenizer.pad_token_id

    llm_head.weight = torch.nn.Parameter(llm_head.weight[unwanted_tokens_mask])
    llm_head.out_features = len(tokenizer) - len(removed_token_ids)

    new_model.lm_head = LearnableBlankLinear(llm_head, new_model.config.blank_token_id)
    new_model.config.ctc_zero_infinity = True

    for module in [
        *new_model.encoder.layers[: int(len(new_model.encoder.layers) * 5 / 6)],
        new_model.encoder.conv1,
        new_model.encoder.conv2,
    ]:
        for param in module.parameters():
            param.requires_grad = False
    return new_model, llm


class LLMASRModel(nn.Module):
    def __init__(self, llm, asr, number_of_prompt_tokens=16, freeze_asr=False, freeze_llm=False):
        super().__init__()
        self.llm = llm
        self.asr = asr
        self.number_of_prompt_tokens = number_of_prompt_tokens
        self.soft_prompt = nn.Embedding(self.number_of_prompt_tokens + 1, llm.config.hidden_size)

        # Initialize soft prompt embeddings
        embeds = self.llm.get_input_embeddings()
        embeds_mean = embeds.weight.mean(dim=0)
        self.soft_prompt.weight.data = embeds_mean.repeat(self.number_of_prompt_tokens + 1, 1)

        if freeze_asr:
            self.freeze_asr()
        if freeze_llm:
            self.freeze_llm()

    def freeze_asr(self):
        for param in self.asr.parameters():
            param.requires_grad = False

    def freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False

    def forward(self, **kwargs):
        labels = kwargs.pop("labels")

        asr_outputs = self.asr(**kwargs)
        asr_predictions = asr_outputs.logits.argmax(dim=-1)
        blank_mask = asr_predictions == self.asr.config.pad_token_id
        deduplication_mask = torch.hstack(
            [
                torch.ones((asr_predictions.shape[0], 1), dtype=torch.bool, device=asr_predictions.device),
                asr_predictions[:, 1:] != asr_predictions[:, :-1],
            ]
        )
        mask = ~blank_mask & deduplication_mask

        soft_prompts = self.soft_prompt(torch.arange(1, self.number_of_prompt_tokens + 1))
        end_prompt = self.soft_prompt(torch.tensor(0)).unsqueeze(0)

        if labels is not None:
            llm_labels = []
            asr_embeddings = []
            for idx, sequence in enumerate(asr_predictions):
                labels_wo_pad = labels[idx][labels[idx] != -100][1:]
                asr_sequence_mask = mask[idx]
                asr_sequence = sequence[mask[idx]].apply_(new_token_ids_mapping_inverted.get)
                asr_sequence_embeds = asr_outputs.hidden_states[idx][asr_sequence_mask]
                asr_embeddings.append(asr_sequence_embeds)
                label_sequence = torch.tensor(
                    (
                        [self.llm.config.bos_token_id]
                        + [self.llm.config.pad_token_id] * (self.number_of_prompt_tokens + len(asr_sequence) + 1)
                        + labels_wo_pad.tolist()
                        + [self.llm.config.eos_token_id]
                    )
                )
                llm_labels.append(label_sequence)
            llm_labels = torch.nn.utils.rnn.pad_sequence(
                llm_labels, batch_first=True, padding_value=self.llm.config.pad_token_id
            )
            input_embeds = self.llm.get_input_embeddings()(llm_labels)
            input_embeds[:, 1 : self.number_of_prompt_tokens + 1] = soft_prompts
            for idx, asr_sequence_embeds in enumerate(asr_embeddings):
                last_asr_embed_position = self.number_of_prompt_tokens + 1 + asr_sequence_embeds.shape[0]
                input_embeds[idx, self.number_of_prompt_tokens + 1 : last_asr_embed_position] = asr_sequence_embeds
                input_embeds[idx, last_asr_embed_position] = end_prompt

            llm_labels[llm_labels == self.llm.config.pad_token_id] = -100
        else:
            llm_labels = None
            input_embeds = []
            bos_token_embed = self.llm.get_input_embeddings()(torch.tensor(tokenizer.bos_token_id))
            pad_token_embed = self.llm.get_input_embeddings()(torch.tensor(tokenizer.pad_token_id))
            for idx, sequence in enumerate(asr_predictions):
                asr_sequence_mask = mask[idx]
                asr_sequence_embeds = asr_outputs.hidden_states[idx][asr_sequence_mask]
                input_embeds.append(
                    torch.vstack([bos_token_embed.unsqueeze(0), soft_prompts, asr_sequence_embeds, end_prompt])
                )
            longest_sequence = max([embeds.shape[0] for embeds in input_embeds])
            for idx, embeds in enumerate(input_embeds):
                input_embeds[idx] = torch.vstack(
                    [embeds, pad_token_embed.repeat(longest_sequence - embeds.shape[0], 1)]
                )
            input_embeds = torch.stack(input_embeds)

        llm_output = self.llm(inputs_embeds=input_embeds, labels=llm_labels)

        return llm_output


if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser(
        (CustomModelArgumentsPrompting, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments)
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

    new_token_ids_mapping, new_token_ids_mapping_inverted, removed_token_ids = get_token_subset(tokenizer)

    # 3. Instantiate model
    asr, llm = get_model(model_args)
    model = LLMASRModel(llm, asr, model_args.number_of_prompt_tokens, model_args.freeze_asr, model_args.freeze_llm)
    model.asr.load_state_dict(safetensors.torch.load_file(model_args.asr_model_checkpoint))

    # 4. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)

    # 5. Initialize data collator
    data_collator = CustomCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        model_input_name=asr.main_input_name,
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
        preprocess_logits_for_metrics=lambda x, y: ctc_greedy_decode(
            x, y, asr.config.blank_token_id, tokenizer.pad_token_id
        ),
        compute_metrics=lambda pred: compute_metrics_ctc(
            tokenizer, new_token_ids_mapping_inverted, pred, gen_args.wandb_predictions_to_save
        ),
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
            token_mapping=new_token_ids_mapping_inverted,
        )

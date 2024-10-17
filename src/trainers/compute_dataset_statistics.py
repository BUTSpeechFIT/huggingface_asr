"""Main training script for training of attention based encoder decoder ASR models."""
import sys
import random

from transformers import AutoFeatureExtractor, HfArgumentParser
from transformers.utils import logging
import torch
from models.bestrq import BestRQModel
from utilities.callbacks import GumbelTemperatureCallback, init_callbacks
from utilities.collators import DataCollatorForWav2Vec2Pretraining
from utilities.data_utils import get_dataset, audio_object_stripper
from utilities.model_utils import instantiate_speech_encoder_model
from utilities.training_arguments import (
    DataTrainingArguments,
    GenerationArguments,
    ModelArguments,
    PretrainingArguments,
)
from utilities.training_utils import SSLTrainer


def process_batch(batch, feature_extractor):
    speech = [audio_object_stripper(sample) for sample in batch]

    # Filter out really small audios which may construct batch from which statistics cannot be estimated.
    filtered_speech = [sample for sample in speech if sample.shape[0] > 10]

    outputs = feature_extractor(filtered_speech, return_attention_mask=True, padding=True, sampling_rate=16000,
                                return_tensors="pt")

    x_active = torch.vstack(
        [x[:mask.sum().item()] for x, mask in zip(outputs["input_features"], outputs["attention_mask"])])
    mean = x_active.mean(dim=0)
    std = x_active.std(dim=0)
    return {"means": mean[None, ...], "stds": std[None, ...]}


from utilities.bind import bind_all

if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((DataTrainingArguments, PretrainingArguments))

    data_args, training_args = parser.parse_args_into_dataclasses()

    # 0. Bind auto classes
    bind_all()

    # 1. Collect, preprocess dataset and extract evaluation dataset
    dataset, training_eval_dataset = get_dataset(
        data_args=data_args,
        len_column=training_args.length_column_name,
    )

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)
    if hasattr(feature_extractor, "do_ceptral_normalize"):
        feature_extractor.do_ceptral_normalize = False

    train_set = dataset[data_args.train_split]
    random.seed(42)
    # Randomly select indices
    if data_args.data_mvn_samples_size > -1:
        indices = random.sample(range(len(train_set)), k=data_args.data_mvn_samples_size)
        train_set = train_set.select(indices)
    
    train_set = train_set.map(
        process_batch,
        batched=True,
        batch_size=training_args.per_device_train_batch_size,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_set.column_names,
        input_columns=["audio"],
        fn_kwargs={"feature_extractor": feature_extractor},
    )
    global_means = torch.tensor(train_set["means"]).mean(dim=0)
    global_stds = torch.tensor(train_set["stds"]).mean(dim=0)

    # Save global means and stds
    torch.save(global_means, f"{training_args.output_dir}/global_means.pt")
    torch.save(global_stds, f"{training_args.output_dir}/global_stds.pt")

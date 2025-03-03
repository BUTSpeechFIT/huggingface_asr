"""Module to preprocess a dataset and save it to disk."""
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
from transformers import HfArgumentParser


@dataclass
class DatasetArguments:
    """Arguments for dataset preprocessing"""

    dataset_builder: str = field(metadata={"help": "The dataset to use."})
    metadata_dir: str = field(metadata={"help": "The directory of the dataset metadata."})
    splits: Optional[List[str]] = field(default=None, metadata={"help": "Dataset splits."})
    output_dir: Optional[str] = field(default=None, metadata={"help": "The directory to save the processed dataset."})
    num_proc: Optional[int] = field(default=1, metadata={"help": "The number of processes to use."})
    regenerate: Optional[bool] = field(default=False, metadata={"help": "Whether to regenerate the dataset."})


if __name__ == "__main__":
    parser = HfArgumentParser((DatasetArguments,))

    (args,) = parser.parse_args_into_dataclasses()

    dataset = datasets.load_dataset(
        args.dataset_builder,
        keep_in_memory=False,
        data_dir=args.metadata_dir,
        splits=args.splits,
        num_proc=args.num_proc,
        download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD
        if args.regenerate
        else datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    print(dataset)
    if args.output_dir is not None:
        dataset.save_to_disk(args.output_dir, num_proc=args.num_proc)

import os
import pickle  # nosec
from typing import Dict, List

import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput


def save_nbests(
    path: str,
    nbests: List[torch.Tensor],
    scores: List[torch.Tensor],
    labels: List[torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    group_size: int = 1,
    batch_size: int = 1,
    outputs: List[Dict] = None,
):
    nbests = [tokenizer.decode(elem.tolist(), skip_special_tokens=True) for item in nbests for elem in item.unbind()]
    processed_labels = []
    if outputs is not None:
        for index, output in enumerate(outputs):
            with open(
                path + f"_utterance{index * batch_size}-{(index + 1) * batch_size - 1}.pkl", "wb"
            ) as file_handler:
                pickle.dump(output, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    for label in labels:
        label[label == -100] = tokenizer.pad_token_id
        processed_labels.extend(
            [
                tokenizer.decode(elem.tolist(), skip_special_tokens=True)
                for elem in label.repeat_interleave(group_size, dim=0)
            ]
        )
    scores = [float(elem) for item in scores for elem in item.unbind()]
    with open(path + "_scores.txt", "w") as file_handler_1:
        with open(path + "_hyps.txt", "w") as file_handler_2:
            with open(path + "_refs.txt", "w") as file_handler_3:
                for item, (sample, score, ref) in enumerate(zip(nbests, scores, processed_labels)):
                    utterance_id = f"utterance{item // group_size}-{item % group_size + 1}"
                    file_handler_1.write(f"{utterance_id} {score}\n")
                    file_handler_2.write(f"{utterance_id} {sample}\n")
                    file_handler_3.write(f"{utterance_id} {ref}\n")


def save_predictions(tokenizer: PreTrainedTokenizer, predictions: PredictionOutput, path: str):
    pred_ids = predictions.predictions

    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    df = pd.DataFrame({"label": label_str, "prediction": pred_str})
    df.to_csv(path, index=False)

    sclite_files = [path.replace(".csv", f"_{type}.trn") for type in ["hyp", "ref"]]
    for strings, file_to_save in zip([pred_str, label_str], sclite_files):
        with open(file_to_save, "w") as file_handler:
            for index, string in enumerate(strings):
                file_handler.write(f"{string} (utterance_{index})\n")

    # evaluate wer also with sclite
    os.system(f"sclite -F -D -i wsj -r {sclite_files[1]} trn -h {sclite_files[0]} trn -o snt sum")  # nosec

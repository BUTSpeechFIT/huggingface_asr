# ELOQUENCE D2.1

## Installation
First, please follow the steps in the main README.md to setup the python environment and PYTHONPATH correctly so that all modules get imported correctly.

## Example DST recipes on SpokenWOZ
In the ```recipes/eloquence/``` directory, you will find a number of example recipes for training and evaluating the models on the SpokenWOZ dataset. Most of these are our current experiment recipes and will get cleaned up once the work finishes.

However, we provide two good starting point recipes for both training and inference in ```recipes/eloquence/examples/```. These recipes contain instructions for setting up your own experiments in terms of modifying crucial paths and experiemnt parameters. The recipes are set up for our own internal cluster, but the core of the recipe should not change across clusters, and once set up properly, they can be also run as standalone experiment scripts.

Currently, these two example recipes utilize the most recent version of the SpokenWOZ trainer (```src/trainers/alignment/train_ecd_lm_woz_new.py```). This trainer and the ```SpeechEncoderConnectorLLM``` aligned model implementation used here  will ultimately replace all previously used versions found in the other trainers. Currently, this trainer only supports DST on SpokenWOZ, but we will extend it to support also our ASR aligned model pretrainining so that you can control the whole training process of the aligned DST models. For now, we will provide all the base ASR checkpoints to be used as initalization points for further DST finetuning.

## SpokenWOZ data preparation
Upon request, we can provide the SpokenWOZ dataset in the Arrow format.

In the other case, SpokenWOZ preparation is very simple. After downloading the dataset from `https://spokenwoz.github.io/SpokenWOZ-github.io/`, use the `recipes/eloquence/prepare_spokenwoz.py` script to convert the dataset to the Arrow format. Lastly, modify the `recipes/eloquence/datasets_spokenwoz.json` file to point to the generated arrow dataset.

## SpokenWOZ DST pre-trained models
For now, we provide two of our best performing models trained on the SpokenWOZ dataset based on WavLM-large, OLMo-1B and Gemma2-9B-it. These models can be found on huggingface:
- `pirxus/wavlm-large_olmo1b_lora_r16a16_np_ua_swft`
- `pirxus/wavlm-large_gemma2-9b-it_fr_nc_lora_r8a8_ua_swft`

For training the `wavlm-large_olmo1b_lora_r16a16_np_ua_swft`, the `phase1_ft_nc` ASR checkpoint from D2.1 was used as the initialization point. It can be found on the ELOQUENCE sharepoint linked down below, in the phase 1 model tarball.

## Loading the models for inference
For inference, it is best to use the example inference recipe ```recipes/eloquence/examples```, as the necessity of sequentially decoding all turns in a dialogue is handled there, and the inference is run on multiple GPUs. The prediction function `do_generate_woz_batched()` can be found in `src/utilities/general_utils.py` and can serve as a useful reference for building your own inference scripts. This inference script will produce a detailed json file in the experiment directory, with all the generated outputs as well as per-turn ASR hypotheses. In case one does not necesarily need on-the-fly inference, running offline inference will save a considerable amount of time (using the 1B model, the inference on SpokenWOZ test takes about 30 minutes on 4 A5500 GPUs).

The models can be loaded using the `SpeechEncoderConnectorLLM` class, which is a wrapper around the Huggingface model. The model can be loaded and used for inference as follows:

```python
import torch
from transformers import AutoTokenizer, AutoFeatureExtractor
from models.aligned_decoder_lm import SpeechEncoderConnectorLLM
from utilities.collators import WOZCollator
from datasets import load_from_disk

# Load the dataset, model, tokenizer and feature extractor
dataset = load_from_disk("path/to/spokenwoz/arrow") # TODO: replace with the correct path
model = SpeechEncoderConnectorLLM.from_pretrained("pirxus/wavlm-large_olmo1b_lora_r16a16_np_ua_swft")
tokenizer = AutoTokenizer.from_pretrained("pirxus/wavlm-large_olmo1b_lora_r16a16_np_ua_swft")
feature_extractor = AutoFeatureExtractor.from_pretrained("pirxus/wavlm-large_olmo1b_lora_r16a16_np_ua_swft")
model = model.cuda()
model.eval()

# Create the collator
data_collator = WOZCollator(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    audio_path='audio',
    text_path='text',
    model_input_name='input_features',
    prompt_prefix='',
)

example = dataset['test'][10]

# The collator will handle the model input preparation if the datasets is in the right format
model_inputs = data_collator([example]).to(model.device)
outputs = model.generate(**model_inputs, generation_config=model.generation_config).cpu()
generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print('{"transcript": ' + generated_batch[0])
"""
'{"transcript": "No, I don\'t think I need email. Thank you.", "domains": ["profile", "restaurant"], "slots": {"profile": {"name": "Kathleen Romaine"}, "restaurant": {"day": "Saturday", "people": "3", "time": "13:10", "area": "West", "food": "Indian"}}}'
"""
```


## D2.1 Pre-trained models

Some of the key pre-trained models described in the deliverable are available in the ELOQUENCE WP2 [sharepoint](https://telefonicacorp.sharepoint.com/:f:/r/sites/ELOQUENCE.TMEHI/Shared%20Documents/WP2/1.%20Deliverables/models_T2.1?csf=1&web=1&e=ycICTb).

There are three archives available in the sharepoint, one for each pre-traininig phase, as described in the deliverable. For phase one, there are two models: FT\_NC and FR\_NC. For phase two, there are three models: two of them come from the FT\_NC checkpoint and the WavLM encoder is either fine-tuned or frozen, and the third model is initialized from the FR\_NC model. For phase three, we provide the single best model trained for joint ASR, user intent recognition, and dialogue slot filling on SLURP.

The individual checkpoints all contain the respective configuration files, which can be used to load the model for inference or further fine-tuning. More detailed information about how the models can be loaded and used for downstream tasks/experiments will be included here in the future. For now, it's best to refer to the recipes in ```recipes/eloquence/```.

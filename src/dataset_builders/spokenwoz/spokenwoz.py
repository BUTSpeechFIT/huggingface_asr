"""Dataset builder module for the SpokenWOZ dataset"""
import sys
import string
import re
import datasets
import json
import pandas as pd
from typing import Optional, List
import os
import re
import torchaudio
from glob import glob
from functools import reduce

TAGS = {
    'user': '0',
    'system': '1',
}

CORRUPTED_TEST = [
    'SNG0601',
    'SNG0646',
    'SNG0653',
    'SNG0877',
    'SNG0885',
    'SNG0890',
    'SNG0897',
    'SNG0901',
    'SNG0903',
]

def shorten_pattern(text):
    """
    Shortens all substrings in the input text that match the pattern
    to a maximum of 20 characters.
    
    Args:
        text (str): Input text containing patterns to be shortened
        
    Returns:
        str: Text with matching patterns shortened to 30 characters
    """
    def replacer(match):
        # Get the matched string
        matched = match.group(0)
        # If the matched string is longer than 20 characters, truncate it
        if len(matched) > 20:
            # Keep the first 19 characters and add a hyphen if the original ended with one
            return matched[:19] + ('-' if matched.endswith('-') else '')
        return matched
    
    pattern = r'(-([a-zA-Z0-9]|(10))){20,}-?'
    
    # Replace all matches using the replacer function
    return re.sub(pattern, replacer, text)

def flatten_dict(metadata):
    ret = {}
    for key, val in metadata.items():
        try:
            val['book'].pop('booked')
        except:
            pass
        ret[key] = val['book'] | val['semi']
    return ret

def minimize_dict(metadata):
    ret = {}
    for key, val in metadata.items():
        vals = {}
        for kk, vv in val.items():
            if vv != "":
                vals[kk] = vv
        if vals != {}:
            ret[key] = vals

    return ret

def get_domains(metadata):
    return list(metadata.keys())


class SpokenWOZ(datasets.GeneratorBasedBuilder):
    """Dataset builder for the raw audio version of the HOW2 dataset"""

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig

    def __init__(self,
                 data_dir: Optional[str] = None,
                 mode: Optional[str] = 'multiwoz',
                 splits: Optional[List[str]] = [],
                 asr_json_dir: Optional[str] = None,
                 title_case_slots: Optional[bool] = True,
                 **kwargs):
        self.mode = mode # has to be in ['multiwoz', 'turn_by_turn', 'collate_user']
        self.data_dir = data_dir
        self.asr_json_dir = asr_json_dir
        self.splits = splits if splits else ['dev', 'train', 'test']
        self.title_case_slots = title_case_slots
        super().__init__(data_dir=data_dir, **kwargs)

    def _info(self):
        if self.mode == 'multiwoz':
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "audio": datasets.Audio(sampling_rate=16_000),
                        "wav_id": datasets.Value('string'),
                        "turn_index": datasets.Value("int32"),
                        "text": datasets.Value("string"),
                        "agent_text": datasets.Value("string"),
                        "domains": datasets.Value('string'),
                        "slots": datasets.Value('string'),
                        # TODO: subsequently convert to something consistent with fisher..
                        "context": datasets.Sequence(feature={
                            "turn_index": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                            "agent_text": datasets.Value("string"),
                            "domains": datasets.Value('string'),
                            "slots": datasets.Value('string'),
                        }),
                    }
                ),
                supervised_keys=None,
            )
        elif self.mode == 'turn_by_turn':
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "audio": datasets.Audio(sampling_rate=16_000),
                        "wav_id": datasets.Value('string'),
                        "turn_index": datasets.Value("int32"),
                        "text": datasets.Value("string"),
                        "text_asr": datasets.Value("string"),
                        "span_info": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value('string'))),
                        "dialog_act": datasets.Value('string'),
                        "metadata": datasets.Value('string'),
                        "tag": datasets.ClassLabel(num_classes=2, names=['user', 'system']),
                        "start_time": datasets.Value("int32"),
                        "end_time": datasets.Value("int32"),
                        # TODO: subsequently convert to something consistent with fisher..
                        "context": datasets.Sequence(feature={
                            "turn_index": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                            "text_asr": datasets.Value("string"),
                            "span_info": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value('string'))),
                            "dialog_act": datasets.Value('string'),
                            "metadata": datasets.Value('string'),
                            "tag": datasets.ClassLabel(num_classes=2, names=['user', 'system']),
                            "start_time": datasets.Value("int32"),
                            "end_time": datasets.Value("int32"),
                        }),
                    }
                ),
                supervised_keys=None,
            )

        else:
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "audio": datasets.Audio(sampling_rate=16_000),
                        "wav_id": datasets.Value('string'),
                        "turn_index": datasets.Value("int32"),
                        "text": datasets.Value("string"),
                        "text_asr": datasets.Value("string"),
                        "agent_text": datasets.Value("string"),
                        "agent_text_asr": datasets.Value("string"),
                        "span_info": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value('string'))),
                        "agent_span_info": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value('string'))),
                        "dialog_act": datasets.Value('string'),
                        "agent_dialog_act": datasets.Value('string'),
                        "metadata": datasets.Value('string'),
                        "start_time": datasets.Value("int32"),
                        "end_time": datasets.Value("int32"),
                        # TODO: subsequently convert to something consistent with fisher..
                        "context": datasets.Sequence(feature={
                            "turn_index": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                            "text_asr": datasets.Value("string"),
                            "agent_text": datasets.Value("string"),
                            "agent_text_asr": datasets.Value("string"),
                            "span_info": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value('string'))),
                            "agent_span_info": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value('string'))),
                            "dialog_act": datasets.Value('string'),
                            "agent_dialog_act": datasets.Value('string'),
                            "metadata": datasets.Value('string'),
                            "start_time": datasets.Value("int32"),
                            "end_time": datasets.Value("int32"),
                        }),
                    }
                ),
                supervised_keys=None,
            )

    def _split_generators(self, _):
        """Generate dataset splits"""
        splits = [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs=self._fetch_split_meta(split),
            )
            for split in self.splits
        ]
        return splits

    def _fetch_split_meta(self, split: str):

        split_dir = 'test' if split == 'test' else 'train_dev'
        with open(str(self.data_dir) + f'/text_5700_{split_dir}/data.json') as json_file:
            json_data = json.load(json_file)

            if split in ['train', 'dev']:
                with open(str(self.data_dir) + f'/text_5700_train_dev/valListFile.json') as val_list:
                    val_file_list = [ line.strip() for line in val_list ]
                
                if split == 'dev':
                    audio_files = { wav: str(self.data_dir) + f'/audio_5700_{split_dir}/' + wav + '.wav' for wav in val_file_list }
                    json_data = dict(filter(lambda pair: pair[0] in val_file_list, json_data.items()))

                else: # train, remove the validation split files
                    audio_files = { wav: str(self.data_dir) + f'/audio_5700_{split_dir}/' + wav + '.wav' for wav in json_data.keys() - val_file_list }

                    for key in val_file_list:
                        json_data.pop(key)
            else:
                audio_files = { wav: str(self.data_dir) + f'/audio_5700_{split_dir}/' + wav + '.wav' for wav in json_data.keys() }

            # finaly construct the feature dict
            features = list(json_data.items())

        asr_transcripts = None
        if self.asr_json_dir:
            asr_jsons = glob(self.asr_json_dir + '/' + split + '/*json')
            asr_transcripts = { json_path.rpartition('/')[2][:-5]: json_path for json_path in asr_jsons }

        return {
            'recordings': audio_files,
            'features': features,
            'asr_transcripts': asr_transcripts,
            'split': split,
        }

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, recordings, features, asr_transcripts, split):
        if self.mode == 'multiwoz':
            for wav_id, data in features:
                if split == 'test' and wav_id in CORRUPTED_TEST: continue # prevent the corrupted test files from being processed

                audio_file = recordings[wav_id]
                if not os.path.isfile(audio_file):
                    continue

                asr_trancript_json = None
                if asr_transcripts:
                    asr_transcript_path = asr_transcripts[wav_id]
                    if not os.path.isfile(asr_transcript_path):
                        raise ValueError(f'This json file does not exist: {asr_transcript_path}')
                    
                    # read json file with path 'asr_transcript'
                    asr_trancript_json = reduce(lambda x, y: {**x, **y}, json.load(open(asr_transcript_path)))

                audio, sr = torchaudio.load(audio_file)
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
                channels = {'user': audio[0], 'system': audio[1]}

                context = []

                def next_log(j):
                    return data['log'][j + 1]

                for i, turn in enumerate(data['log']):
                    if i % 2 == 1: continue
                    agent_dialog_act = json.dumps(next_log(i)['dialog_act'])
                    metadata = next_log(i)['metadata']
                    slots = minimize_dict(flatten_dict(metadata))
                    domains = get_domains(slots)
                    tag = turn['tag']
                    text = turn['text']
                    agent_text = next_log(i)['text']
                    
                    # get the corresponding audio slice
                    start_time = turn['words'][0]['BeginTime']
                    end_time = turn['words'][-1]['EndTime']
                    audio_slice = channels[tag][start_time*16:end_time*16]

                    # get the asr transcript from the json file if available
                    if asr_transcripts:
                        text = asr_trancript_json['turn_id_' + str(i) + '_tag_0']['text']
                        agent_text = asr_trancript_json['turn_id_' + str(i + 1) + '_tag_1']['text']

                        text = shorten_pattern(text)
                        agent_text = shorten_pattern(agent_text)
                    else:
                        # preprocess the text, remove punctuation
                        def process_text(txt):
                            txt = txt.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
                            txt = re.sub(' +', ' ', txt)
                            txt = txt.strip()
                            return txt
                        text = process_text(text)
                        agent_text = process_text(agent_text)


                    return_dict = {
                        'audio': datasets.features.Audio(sampling_rate=16000).encode_example({
                                'path': None,
                                'array': audio_slice,
                                'sampling_rate': 16000,
                        }),
                        'wav_id': wav_id,
                        'turn_index': i,
                        'text': text,
                        'agent_text': agent_text,
                        'domains': domains,
                        'slots': slots,
                        'context': context,
                    }

                    if self.title_case_slots:
                        for domain, slot in return_dict['slots'].items():
                            for slot_key, slot_value in slot.items():
                                if slot_key in ("destination", "day", "departure", "food", "name"):
                                    return_dict['slots'][domain][slot_key] = slot_value.title()

                                elif slot_key in ("area") and slot_value in ("east", "west", "north", "south"):
                                    return_dict['slots'][domain][slot_key] = slot_value.title()

                    yield wav_id + '_' + str(i), return_dict

                    return_dict.pop('wav_id')
                    return_dict.pop('context')
                    return_dict.pop('audio')
                    context.append(return_dict)

        elif self.mode == 'turn_by_turn':
            for wav_id, data in features:
                if split == 'test' and wav_id in CORRUPTED_TEST: continue # prevent the corrupted test files from being processed

                audio_file = recordings[wav_id]
                if not os.path.isfile(audio_file):
                    continue

                asr_trancript_json = None
                if asr_transcripts:
                    asr_transcript_path = asr_transcripts[wav_id]
                    if not os.path.isfile(asr_transcript_path):
                        raise ValueError(f'This json file does not exist: {asr_transcript_path}')
                    
                    # read json file with path 'asr_transcript'
                    asr_trancript_json = reduce(lambda x, y: {**x, **y}, json.load(open(asr_transcript_path)))

                audio, sr = torchaudio.load(audio_file)
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
                channels = {'user': audio[0], 'system': audio[1]}

                context = []

                for i, turn in enumerate(data['log']):
                    span_info = turn['span_info']
                    dialog_act = json.dumps(turn['dialog_act'])
                    metadata = json.dumps(turn['metadata'])
                    tag = turn['tag']
                    text = turn['text']
                    start_time = turn['words'][0]['BeginTime']
                    end_time = turn['words'][-1]['EndTime']
                    
                    # get the corresponding audio slice
                    audio_slice = channels[tag][start_time*16:end_time*16]

                    # preprocess the text
                    # remove punctuation
                    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
                    text = re.sub(' +', ' ', text)
                    text = text.strip()

                    text_asr = None
                    # get the asr trasncript from the json file if available
                    if asr_transcripts:
                        text_asr = asr_trancript_json['turn_id_' + str(i) + '_tag_' + TAGS[str(tag)]]['text']

                    return_dict = {
                        'audio': datasets.features.Audio(sampling_rate=16000).encode_example({
                                'path': None,
                                'array': audio_slice,
                                'sampling_rate': 16000,
                        }),
                        'wav_id': wav_id,
                        'turn_index': i,
                        'text': text,
                        'text_asr': text_asr,
                        'span_info': span_info,
                        'dialog_act': dialog_act,
                        'metadata': metadata,
                        'tag': tag,
                        'start_time': start_time,
                        'end_time': end_time,
                        'context': context,
                    }

                    yield wav_id + '_' + str(i), return_dict

                    return_dict.pop('wav_id')
                    return_dict.pop('context')
                    return_dict.pop('audio')
                    context.append(return_dict)

        else:
            for wav_id, data in features:
                if split == 'test' and wav_id in CORRUPTED_TEST: continue # prevent the corrupted test files from being processed

                audio_file = recordings[wav_id]
                if not os.path.isfile(audio_file):
                    continue

                asr_trancript_json = None
                if asr_transcripts:
                    asr_transcript_path = asr_transcripts[wav_id]
                    if not os.path.isfile(asr_transcript_path):
                        raise ValueError(f'This json file does not exist: {asr_transcript_path}')
                    
                    # read json file with path 'asr_transcript'
                    asr_trancript_json = reduce(lambda x, y: {**x, **y}, json.load(open(asr_transcript_path)))

                audio, sr = torchaudio.load(audio_file)
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
                channels = {'user': audio[0], 'system': audio[1]}

                context = []

                def next_log(j):
                    return data['log'][j + 1]

                for i, turn in enumerate(data['log']):
                    if i % 2 == 1: continue
                    span_info = turn['span_info']
                    agent_span_info = next_log(i)['span_info']
                    dialog_act = json.dumps(turn['dialog_act'])
                    agent_dialog_act = json.dumps(next_log(i)['dialog_act'])
                    metadata = json.dumps(next_log(i)['metadata'])
                    tag = turn['tag']
                    text = turn['text']
                    agent_text = next_log(i)['text']
                    start_time = turn['words'][0]['BeginTime']
                    end_time = turn['words'][-1]['EndTime']
                    
                    # get the corresponding audio slice
                    audio_slice = channels[tag][start_time*16:end_time*16]

                    # preprocess the text
                    # remove punctuation
                    def process_text(txt):
                        txt = txt.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
                        txt = re.sub(' +', ' ', txt)
                        txt = txt.strip()
                        return txt
                    text = process_text(text)
                    agent_text = process_text(agent_text)

                    text_asr = None
                    agent_text_asr = None
                    # get the asr trasncript from the json file if available
                    if asr_transcripts:
                        text_asr = asr_trancript_json['turn_id_' + str(i) + '_tag_0']['text']
                        agent_text_asr = asr_trancript_json['turn_id_' + str(i + 1) + '_tag_1']['text']

                    return_dict = {
                        'audio': datasets.features.Audio(sampling_rate=16000).encode_example({
                                'path': None,
                                'array': audio_slice,
                                'sampling_rate': 16000,
                        }),
                        'wav_id': wav_id,
                        'turn_index': i,
                        'text': text,
                        'text_asr': text_asr,
                        'agent_text': agent_text,
                        'agent_text_asr': agent_text_asr,
                        'span_info': span_info,
                        'agent_span_info': agent_span_info,
                        'dialog_act': dialog_act,
                        'agent_dialog_act': agent_dialog_act,
                        'metadata': metadata,
                        'tag': tag,
                        'start_time': start_time,
                        'end_time': end_time,
                        'context': context,
                    }

                    yield wav_id + '_' + str(i), return_dict

                    return_dict.pop('wav_id')
                    return_dict.pop('context')
                    return_dict.pop('audio')
                    context.append(return_dict)

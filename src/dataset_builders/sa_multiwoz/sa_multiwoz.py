"""Dataset builder module for the Speech-aware MultiWOZ dataset"""
import datasets
import json
from typing import Optional, List
import os
import torchaudio
import random

class SpeechAwareMultiWOZ(datasets.GeneratorBasedBuilder):
    """Dataset builder module for the Speech-aware MultiWOZ dataset

    This builder expects a MultiWOZ dataset directory of our own specific format,
    it is not usable with the original MultiWOZ dataset.
    """

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig

    def __init__(self,
                 data_dir: Optional[str] = None,
                 splits: Optional[List[str]] = [],
                 title_case_slots: Optional[bool] = True,
                 **kwargs):
        self.splits = splits if splits else ['dev', 'train']
        self.data_dir = data_dir
        self.title_case_slots = title_case_slots
        super().__init__(data_dir=data_dir, **kwargs)

    def _info(self):
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

        with open(str(self.data_dir) + f'/{split}_punct_v2.json') as json_file:
            conversations = [ json.loads(line) for line in json_file ]

        return {
            'features': conversations,
        }

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, features):
        random.seed(42)

        for conversation in features:
            wav_id = list(conversation.keys())[0]
            metadata = conversation[wav_id]['log']

            def next_turn(j):
                try: return metadata[j + 1]
                except: return None

            # reset the context for each conversation
            context = []
            for i, turn in enumerate(metadata):
                if i % 2 == 1: continue

                # make radom choice between the audio files
                audio_file = random.choice(turn['audio'][0])
                audio, sr = torchaudio.load(audio_file)
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)[0]

                nt = next_turn(i)

                return_dict = {
                    'audio': datasets.features.Audio(sampling_rate=16000).encode_example({
                            'path': audio_file,
                            'array': audio,
                            'sampling_rate': 16000,
                    }) if audio is not None else None,
                    'wav_id': wav_id,
                    'turn_index': i,
                    'text': turn['text'],
                    'agent_text': nt['text'] if nt is not None else None,
                    'domains': turn['scenario'],
                    'slots': turn['slot'],
                    'context': context,
                }

                if self.title_case_slots:
                    for domain, slot in return_dict['slots'].items():
                        for slot_key, slot_value in slot.items():
                            if slot_key in ("destination", "day", "departure", "food", "name"):
                                return_dict['slots'][domain][slot_key] = slot_value.title()

                            elif slot_key in ("area") and slot_value in ("east", "west", "north", "south"):
                                return_dict['slots'][domain][slot_key] = slot_value.title()

                if audio is not None:
                    yield wav_id + '_' + str(i), return_dict

                return_dict.pop('wav_id')
                return_dict.pop('context')
                return_dict.pop('audio')
                context.append(return_dict)

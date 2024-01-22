"""Kaldi dataset builder"""
import io
import logging
import math
import os
import wave
from itertools import groupby
from typing import Iterable, List, Optional, Tuple, Union
from transformers.utils import logging

import datasets
import kaldiio
import librosa
import numpy as np

_FILEPATHS = {
    "audios": "wav.scp",
    "segments": "segments",
    "transcripts": "text",
    "segments2speakers": "utt2spk", # not used
    "speakers2segments": "spk2utt", # not used
}

_DATASET_TYPE = {
    "audio_only": 1,     # expects only wav.scp and segments Kaldi files to be defined
    "text_only": 2,      # expects only text Kaldi file to be defined
    "dialog": 3,         # not used - expects only text, spk2utt, utt2spk Kaldi files to be defined
    "full": 4,           # expects all Kaldi files to be defined
}


class KaldiDataset(datasets.GeneratorBasedBuilder):
    """Dataset builder for Kaldi format dataset"""

    DEFAULT_WRITER_BATCH_SIZE = 50  # the default size of the batch may not fit in memory

    def __init__(self, data_dir: Optional[str], splits: List[str], sampling_rate: int = 16_000, data_type: str = "full", **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        self.sampling_rate = sampling_rate
        self.data_dir = data_dir
        self.splits = splits
        self.data_type = _DATASET_TYPE[data_type]
        self.logger = logging.get_logger("transformers")
        if self.data_type == _DATASET_TYPE["dialog"]:
            self.logger.critical("Kaldi data type 'dialog' is not implemented.")

    def _info(self):
        match self.data_type:
            case _DATASET_TYPE["full"]:
                return datasets.DatasetInfo(
                    features=datasets.Features(
                        {
                            "audio": datasets.Audio(sampling_rate=16_000),
                            "labels": datasets.Value("string"),
                            "uttid": datasets.Value("string"),
                            "recording": datasets.Value("string"),
                            "turn_index": datasets.Value("int32"),
                            "input_len": datasets.Value("float32"),
                        }
                    ),
                    supervised_keys=None,
                )
            case _DATASET_TYPE["text_only"]:
                return datasets.DatasetInfo(
                    features=datasets.Features(
                        {
                            "labels": datasets.Value("string"),
                            "uttid": datasets.Value("string"),
                        }
                    ),
                    supervised_keys=None,
                )
            case _DATASET_TYPE["audio_only"]:
                return datasets.DatasetInfo(
                    features=datasets.Features(
                        {
                            "audio": datasets.Audio(sampling_rate=16_000),
                            "uttid": datasets.Value("string"),
                            "recording": datasets.Value("string"),
                            "turn_index": datasets.Value("int32"),
                            "input_len": datasets.Value("float32"),
                        }
                    ),
                    supervised_keys=None,
                )
            case _:
                self.logger.critical("Kaldi data type '%i' is not implemented." % self.data_type)
                return None

    def _prepare_split_single(
        self,
        gen_kwargs: dict,
        fpath: str,
        file_format: str,
        max_shard_size: int,
        split_info: datasets.SplitInfo,
        check_duplicate_keys: bool,
        job_id: int,
    ) -> Iterable[Tuple[int, bool, Union[int, tuple]]]:
        return super()._prepare_split_single(
            gen_kwargs, fpath, file_format, max_shard_size, split_info, check_duplicate_keys, job_id
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

    @staticmethod
    def _split_text_string(text):
        """Split text string into uttid and transcript"""
        parts = text.strip().split(maxsplit=1)
        if len(parts) == 1:
            parts.append("")
        return parts

    def _fetch_split_meta(self, split: str):
        """Fetch split meta data from kaldi-like dataset"""

        text_file = os.path.join(self.data_dir, split, _FILEPATHS["transcripts"])
        segm_file = os.path.join(self.data_dir, split, _FILEPATHS["segments"])
        feat_file = os.path.join(self.data_dir, split, _FILEPATHS["audios"])


        if os.path.exists(text_file):
            with open(text_file) as file:
                texts = dict(map(lambda line: self._split_text_string(line), file))  # creates (segment_id -> text) mapping
        else:
            self.logger.warning("Segment file %s does not exists. Creating a dummy segmentation 0 -> -1." % segm_file)

        segm_file = os.path.join(self.data_dir, split, _FILEPATHS["segments"])
        if os.path.exists(segm_file):
            with open(segm_file) as file:
                segments = dict(map(lambda s: self._parse_segment_info(*s.strip().split()), file))
        else:
            """If segments file does not exist, create dummy mapping (segment_id -> (segment_id, 0, -1))"""
            self.logger.warning("Segment file %s does not exists. Creating a dummy segmentation 0 -> -1." % segm_file)
            segments = dict(map(lambda s: (s, (s, 0, -1)), texts.keys()))

        # load kaldiio feature generator
        feat_file = os.path.join(self.data_dir, split, _FILEPATHS["audios"])
        feats_generator = kaldiio.load_scp(feat_file)
        segments = [(*segments[uttid], uttid, transcript) for (uttid, transcript) in texts.items()]
        grouped_by_recordings = [(k, list(v)) for k, v in groupby(sorted(segments), key=lambda segment: segment[0])]
        return {
            "recordings": grouped_by_recordings,
            "features": feats_generator,
        }

    @staticmethod
    def create_wav_bytes(audio_samples, sample_rate=16000):
        # Ensure the audio samples are in the correct format (16-bit PCM)
        audio_samples = np.int16(audio_samples * 32767)

        # Create a WAV file in-memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "w") as wav_file:
            # Set the WAV file parameters
            wav_file: wave.Wave_write
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(sample_rate)
            wav_file.setnframes(len(audio_samples))
            wav_file.writeframes(audio_samples.tobytes())

        # Get the WAV file content as bytes
        wav_bytes = wav_buffer.getvalue()
        return wav_bytes

    def _generate_examples(self, recordings, features):
        """Generator for split examples fetching"""
        for recording, segments in recordings:
            sampling_rate, audio = features[recording]
            if audio.dtype == np.int16:
                audio = librosa.util.buf_to_float(audio, n_bytes=audio.dtype.itemsize)
            else:
                raise ValueError("Data type of input audio is not int16.")
            if len(audio.shape) > 1:
                raise ValueError(f"Recording {recording} does not have single channel.")
            if sampling_rate != self.sampling_rate or len(audio.shape) > 1:
                logging.debug(f"Resampled {recording} from {sampling_rate} to {self.sampling_rate}")
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=self.sampling_rate)
            sorted_segments = sorted(segments, key=lambda x: x[1])
            for index, (_, start, end, uttid, transcript) in enumerate(sorted_segments):
                audio_cropped = self._crop_audio(audio, self.sampling_rate, start, end)
                text = self.preprocess_text(transcript)
                yield f"{recording}_{index}", {
                    "audio": {"path": None, "bytes": self.create_wav_bytes(audio_cropped)},
                    "labels": text,
                    "uttid": uttid,
                    "recording": recording,
                    "turn_index": index,
                    "input_len": len(audio_cropped) / self.sampling_rate,
                }

    @staticmethod
    def _parse_segment_info(segment_key, uri, start, end):
        """Parse segment info"""
        return segment_key, (uri, float(start), float(end))

    @staticmethod
    def _crop_audio(audio, sampling_rate, start, end):
        """Crop audio"""
        return audio[math.floor(sampling_rate * start) : math.ceil(end * sampling_rate) if end != -1 else end].copy()

    @staticmethod
    def preprocess_text(utterance_batch: List[str]):
        """Preprocess text"""
        return utterance_batch

"""AudioFolderVAD dataset."""
from typing import TYPE_CHECKING, List, Optional, Union

import datasets
import torch
import torchaudio
from datasets.packaged_modules.folder_based_builder import folder_based_builder
from datasets.splits import SplitGenerator
from datasets.tasks import AudioClassification

# pylint: disable=no-name-in-module
from multiprocess import set_start_method
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

if TYPE_CHECKING:
    pass
logger = datasets.utils.logging.get_logger(__name__)


class AudioFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """Builder Config for AudioFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None


class AudioFolderVAD(folder_based_builder.FolderBasedBuilder):
    """AudioFolderVAD dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 100

    BASE_FEATURE = datasets.Audio
    BASE_COLUMN_NAME = "audio"
    BUILDER_CONFIG_CLASS = AudioFolderConfig
    EXTENSIONS: List[str]  # definition at the bottom of the script
    CLASSIFICATION_TASK = AudioClassification(audio_column="audio", label_column="label")

    def __init__(
        self,
        *args,
        vad_model: str = "pyannote/segmentation-3.0",
        vad_device: str = "cpu",
        vad_batch_size: int = 1024,
        vad_min_duration_on: float = 0.0,
        vad_min_duration_off: float = 0.0,
        sampling_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        device = torch.device(vad_device)
        model = Model.from_pretrained(vad_model, use_auth_token=kwargs.get("use_auth_token", None))
        self.vad_pipeline = VoiceActivityDetection(segmentation=model, batch_size=vad_batch_size, device=device)
        params = {
            # remove speech regions shorter than that many seconds.
            "min_duration_on": vad_min_duration_on,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": vad_min_duration_off,
        }
        self.vad_pipeline.instantiate(params)
        self.sampling_rate = sampling_rate

    def _split_generators(self, dl_manager):
        splits = super()._split_generators(dl_manager)
        self.info.features = datasets.Features(**self.info.features, input_len=datasets.Value("float"))
        return splits

    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        check_duplicate_keys: bool,
        file_format="arrow",
        num_proc: Optional[int] = None,
        max_shard_size: Optional[Union[int, str]] = None,
    ):
        set_start_method("spawn")
        super()._prepare_split(split_generator, check_duplicate_keys, file_format, num_proc, max_shard_size)

    def _generate_examples(self, files, metadata_files, split_name, add_metadata, add_labels):
        audio_encoder = datasets.Audio(sampling_rate=self.sampling_rate, mono=True)
        for example_id, example in super()._generate_examples(
            files, metadata_files, split_name, add_metadata, add_labels
        ):
            # pylint: disable=no-member
            waveform, sample_rate = torchaudio.load(example["audio"])

            annotation = self.vad_pipeline({"waveform": waveform, "sample_rate": sample_rate})

            for segment in annotation.itersegments():
                chunk = waveform[:, int(segment.start * sample_rate) : int(segment.end * sample_rate)].squeeze().numpy()
                yield f"{example_id}_{segment.start:.2f}_{segment.end:.2f}", {
                    **example,
                    "audio": audio_encoder.encode_example({"array": chunk, "sampling_rate": sample_rate}),
                    "input_len": len(chunk) / self.sampling_rate,
                }


# Obtained with:
# ```
# import soundfile as sf
#
# AUDIO_EXTENSIONS = [f".{format.lower()}" for format in sf.available_formats().keys()]
#
# # .mp3 is currently decoded via `torchaudio`, .opus decoding is supported if version of `libsndfile` >= 1.0.30:
# AUDIO_EXTENSIONS.extend([".mp3", ".opus"])
# ```
# We intentionally do not run this code on launch because:
# (1) Soundfile is an optional dependency, so importing it in global namespace is not allowed
# (2) To ensure the list of supported extensions is deterministic
AUDIO_EXTENSIONS = [
    ".aiff",
    ".au",
    ".avr",
    ".caf",
    ".flac",
    ".htk",
    ".svx",
    ".mat4",
    ".mat5",
    ".mpc2k",
    ".ogg",
    ".paf",
    ".pvf",
    ".raw",
    ".rf64",
    ".sd2",
    ".sds",
    ".ircam",
    ".voc",
    ".w64",
    ".wav",
    ".nist",
    ".wavex",
    ".wve",
    ".xi",
    ".mp3",
    ".opus",
]
AudioFolderVAD.EXTENSIONS = AUDIO_EXTENSIONS

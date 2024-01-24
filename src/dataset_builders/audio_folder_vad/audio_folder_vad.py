"""AudioFolderVAD dataset."""
from typing import List

import datasets
import torchaudio
from datasets.packaged_modules.folder_based_builder import folder_based_builder
from datasets.tasks import AudioClassification
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

logger = datasets.utils.logging.get_logger(__name__)


class AudioFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """Builder Config for AudioFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None


class AudioFolderVAD(folder_based_builder.FolderBasedBuilder):
    """AudioFolderVAD dataset."""

    BASE_FEATURE = datasets.Audio
    BASE_COLUMN_NAME = "audio"
    BUILDER_CONFIG_CLASS = AudioFolderConfig
    EXTENSIONS: List[str]  # definition at the bottom of the script
    CLASSIFICATION_TASK = AudioClassification(audio_column="audio", label_column="label")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=kwargs.get("use_auth_token", None))
        self.vad_pipeline = VoiceActivityDetection(segmentation=model)
        HYPER_PARAMETERS = {
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.0,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.0,
        }
        self.vad_pipeline.instantiate(HYPER_PARAMETERS)

    def _generate_examples(self, files, metadata_files, split_name, add_metadata, add_labels):
        audio_encoder = datasets.Audio(sampling_rate=16000, mono=True)
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

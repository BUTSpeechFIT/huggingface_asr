# pylint: skip-file
"""SpecAugment implementation copied from ESPnet toolkit.
https://espnet.github.io/espnet/_modules/espnet2/asr/specaug/specaug.html"""
from typing import Union, List, Optional

import torch
from torch_audiomentations import AddBackgroundNoise, ApplyImpulseResponse

class TorchAudiomentations_BackgroundNoise_Wrapper(AddBackgroundNoise):
    """Wrapper for torch-audiomentations/torch_audiomentations/augmentations/background_noise.py

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.unsqueeze(0).unsqueeze(0)).squeeze(0)


class TorchAudiomentations_Reverberation_Wrapper(ApplyImpulseResponse):
    """Wrapper for torch-audiomentations/torch_audiomentations/augmentations/impulse_response.py

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.unsqueeze(0).unsqueeze(0)).squeeze(0)

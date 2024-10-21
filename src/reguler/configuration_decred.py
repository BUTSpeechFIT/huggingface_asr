from transformers import AutoConfig, AutoModelForCausalLM, SpeechEncoderDecoderConfig

from .auto_wrappers import CustomAutoModelForCTC
from .e_branchformer import Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForCTC
from .multi_head_gpt2 import GPT2LMMultiHeadModel, GPT2MultiHeadConfig
from .residual_clasiffier_gpt2 import (
    GPT2ResidualsLMHeadConfig,
    GPT2ResidualsLMHeadModel,
)

AutoConfig.register("gpt2-multi-head", GPT2MultiHeadConfig)
AutoModelForCausalLM.register(GPT2MultiHeadConfig, GPT2LMMultiHeadModel)

AutoConfig.register("gpt2-residuals-head", GPT2ResidualsLMHeadConfig)
AutoModelForCausalLM.register(GPT2ResidualsLMHeadConfig, GPT2ResidualsLMHeadModel)

AutoConfig.register("wav2vec2-ebranchformer", Wav2Vec2EBranchformerConfig)
CustomAutoModelForCTC.register(Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForCTC)


class JointCTCAttentionEncoderDecoderConfig(SpeechEncoderDecoderConfig):
    model_type = "joint_aed_ctc_speech-encoder-decoder"
    is_composition = True

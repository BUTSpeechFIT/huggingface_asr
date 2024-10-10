from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoModelForPreTraining,
    AutoModelForSpeechSeq2Seq,
    AutoFeatureExtractor
)

from models.auto_wrappers import CustomModelForCausalLM
from models.bestrq import (
    BestRQEBranchformerForCTC,
    BestRQEBranchformerForPreTraining,
    BestRQEBranchformerForPreTrainingConfig,
)
from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoder,
    JointCTCAttentionEncoderDecoderConfig,
)
from models.decoders.multi_head_gpt2 import GPT2LMMultiHeadModel, GPT2MultiHeadConfig
from models.decoders.multi_head_gpt2_mixing import (
    GPT2LMMultiHeadModelMixing,
    GPT2MultiHeadMixingConfig,
)
from models.decoders.residual_clasiffier_gpt2 import (
    GPT2ResidualsLMHeadConfig,
    GPT2ResidualsLMHeadModel,
)
from models.encoders.e_branchformer import (
    Wav2Vec2EBranchformerConfig,
    Wav2Vec2EBranchformerForCTC,
    Wav2Vec2EBranchformerForPreTraining,
)
from utilities.feature_extractors import CustomFeatureExtractor, CustomFeatureExtractorConfig


def bind_all():
    AutoConfig.register("joint_aed_ctc_speech-encoder-decoder", JointCTCAttentionEncoderDecoderConfig)
    AutoModelForSpeechSeq2Seq.register(JointCTCAttentionEncoderDecoderConfig, JointCTCAttentionEncoderDecoder)

    AutoConfig.register("wav2vec2-ebranchformer", Wav2Vec2EBranchformerConfig)
    AutoModelForCTC.register(Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForCTC)
    AutoModelForPreTraining.register(Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForPreTraining)

    AutoConfig.register("bestrq-ebranchformer", BestRQEBranchformerForPreTrainingConfig)
    AutoModelForCTC.register(BestRQEBranchformerForPreTrainingConfig, BestRQEBranchformerForCTC)
    AutoModelForPreTraining.register(BestRQEBranchformerForPreTrainingConfig, BestRQEBranchformerForPreTraining)

    AutoConfig.register("gpt2-multi-head", GPT2MultiHeadConfig)
    CustomModelForCausalLM.register(GPT2MultiHeadConfig, GPT2LMMultiHeadModel)

    AutoConfig.register("gpt2-multi-head-mixing", GPT2MultiHeadMixingConfig)
    CustomModelForCausalLM.register(GPT2MultiHeadMixingConfig, GPT2LMMultiHeadModelMixing)

    AutoConfig.register("gpt2-residuals-head", GPT2ResidualsLMHeadConfig)
    CustomModelForCausalLM.register(GPT2ResidualsLMHeadConfig, GPT2ResidualsLMHeadModel)

    AutoConfig.register("custom_feature_extractor", CustomFeatureExtractorConfig)
    AutoFeatureExtractor.register(CustomFeatureExtractorConfig, CustomFeatureExtractor)

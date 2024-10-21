from transformers import AutoFeatureExtractor, AutoTokenizer

from reguler.configuration_decred import JointCTCAttentionEncoderDecoderConfig
from reguler.generation import GenerationConfigCustom
from reguler.modeling_decred import JointCTCAttentionEncoderDecoder

if __name__ == "__main__":
    JointCTCAttentionEncoderDecoderConfig.register_for_auto_class()
    JointCTCAttentionEncoderDecoder.register_for_auto_class("AutoModelForSpeechSeq2Seq")
    model = JointCTCAttentionEncoderDecoder.from_pretrained(
        "/mnt/matylda5/ipoloka/IS24_DeCRED/multidomain/normalised_data/DeCRED_base/checkpoint-231248"
    )

    tokenizer = AutoTokenizer.from_pretrained("Lakoc/english_corpus_uni5000_normalized")
    model.config.tokenizer_class = str(type(tokenizer))
    feature_extractor = AutoFeatureExtractor.from_pretrained("Lakoc/log_80mel_extractor_16k")

    generation_config = model.generation_config
    gen_config = GenerationConfigCustom(**generation_config.to_dict())
    generation_config.ctc_weight = 0.3
    generation_config.num_beams = 5
    generation_config.ctc_margin = 0

    generation_config.push_to_hub("BUT-FIT/DeCRED-base", private=True)
    feature_extractor.push_to_hub("BUT-FIT/DeCRED-base", private=True)
    tokenizer.push_to_hub("BUT-FIT/DeCRED-base", private=True)
    model.push_to_hub("BUT-FIT/DeCRED-base", private=True)

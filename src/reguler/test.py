if __name__ == "__main__":
    # from reguler.configuration_reguler import JointCTCAttentionEncoderDecoderConfig
    # from reguler.modeling_reguler import JointCTCAttentionEncoderDecoder
    #
    # JointCTCAttentionEncoderDecoderConfig.register_for_auto_class()
    # JointCTCAttentionEncoderDecoder.register_for_auto_class("AutoModelForSpeechSeq2Seq")
    # model = JointCTCAttentionEncoderDecoder.from_pretrained(
    #     "/Users/alexanderpolok/PycharmProjects/huggingface_asr/checkpoint-378950"
    # )
    # model.push_to_hub("BUT-FIT/EBranchRegulaFormer-medium")

    from transformers import pipeline

    model_id = "BUT-FIT/EBranchRegulaFormer-medium"
    pipe = pipeline("automatic-speech-recognition", model=model_id, feature_extractor=model_id, trust_remote_code=True)
    # In newer versions of transformers (>4.31.0), there is a bug in the pipeline inference type.
    # The warning can be ignored.
    pipe.type = "seq2seq"

    # Standard greedy decoding
    result = pipe("audio.wav")

    # Beam search decoding with joint CTC-attention scorer
    generation_config = pipe.model.generation_config
    generation_config.ctc_weight = 0.5
    generation_config.num_beams = 5
    generation_config.ctc_margin = 0
    result = pipe("audio.wav")

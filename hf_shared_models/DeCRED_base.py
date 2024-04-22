if __name__ == "__main__":
    from reguler.configuration_decred import JointCTCAttentionEncoderDecoderConfig
    from reguler.modeling_decred import JointCTCAttentionEncoderDecoder

    JointCTCAttentionEncoderDecoderConfig.register_for_auto_class()
    JointCTCAttentionEncoderDecoder.register_for_auto_class("AutoModelForSpeechSeq2Seq")
    model = JointCTCAttentionEncoderDecoder.from_pretrained(
        "/mnt/matylda5/ipoloka/IS24_DeCRED/multidomain/normalised_data/DeCRED_base/checkpoint-231248"
    )
    model.push_to_hub("BUT-FIT/DeCRED-base", private=True)

from transformers import pipeline

if __name__ == "__main__":
    model_id = "BUT-FIT/DeCRED-base"
    pipe = pipeline("automatic-speech-recognition", model=model_id, feature_extractor=model_id, trust_remote_code=True)
    # In newer versions of transformers (>4.31.0), there is a bug in the pipeline inference type.
    # The warning can be ignored.
    pipe.type = "seq2seq"

    # Run beam search decoding with joint CTC-attention scorer
    result_beam = pipe("audio.wav")

    # Run greedy decoding without joint CTC-attention scorer
    pipe.model.generation_config.ctc_weight = 0.0
    pipe.model.generation_config.num_beams = 1

    result_greedy = pipe("audio.wav")

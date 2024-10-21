from typing import Any, Dict, List

from transformers import AutoTokenizer, pipeline

from .modeling_reguler import JointCTCAttentionEncoderDecoder


class EndpointHandler:
    def __init__(self, path=""):
        # load the optimized model
        model = JointCTCAttentionEncoderDecoder.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        # create inference pipeline
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model="Lakoc/reguler_medium",
            tokenizer="Lakoc/english_corpus_uni5000",
            feature_extractor="Lakoc/log_80mel_extractor_16k",
        )

    def __call__(self, data: Any) -> List[List[Dict[str, float]]]:
        """
        Args:
            data (:obj:):
                includes the input data and the parameters for the inference.
        Return:
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing :
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # pass inputs with all kwargs in data
        if parameters is not None:
            prediction = self.pipeline(inputs, **parameters)
        else:
            prediction = self.pipeline(inputs)
        # postprocess the prediction
        return prediction

from transformers import GenerationConfig


class GenerationConfigCustom(GenerationConfig):
    def __init__(
        self,
        ctc_weight=0.0,
        ctc_margin=0,
        lm_weight=0,
        lm_model=None,
        space_token_id=-1,
        eos_space_trick_weight=0,
        apply_eos_space_trick=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctc_weight = ctc_weight
        self.ctc_margin = ctc_margin
        self.lm_weight = lm_weight
        self.lm_model = lm_model
        self.space_token_id = space_token_id
        self.eos_space_trick_weight = eos_space_trick_weight
        self.apply_eos_space_trick = apply_eos_space_trick

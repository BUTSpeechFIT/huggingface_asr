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

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(";"))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise ValueError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)

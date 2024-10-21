from models.bestrq import (  # BestRQTransformerForPreTraining, BestRQTransformerForPreTrainingConfig, \
    BestRQEBranchformerForPreTraining,
    BestRQEBranchformerForPreTrainingConfig,
)
from models.encoders.e_branchformer import (
    Wav2Vec2EBranchformerConfig,
    Wav2Vec2EBranchformerModel,
)

conf = {
    "activation_dropout": 0.0,
    "apply_spec_augment": True,
    "attention_dropout": 0.0,
    "classifier_proj_size": 256,
    "codevector_dim": 256,
    "contrastive_logits_temperature": 0.1,
    "conv_bias": True,
    "conv_dim": [256, 256, 256, 256, 256, 256, 256],
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "ctc_loss_reduction": "sum",
    "ctc_zero_infinity": False,
    "diversity_loss_weight": 0.1,
    "do_stable_layer_norm": True,
    "feat_extract_activation": "gelu",
    "feat_extract_dropout": 0.0,
    "feat_extract_norm": "layer",
    "feat_proj_dropout": 0.0,
    "feat_quantizer_dropout": 0.0,
    "final_dropout": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout": 0.0,
    "hidden_dropout_prob": 0.0,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "layer_norm_eps": 1e-05,
    "layerdrop": 0.0,
    "mask_feature_length": 10,
    "mask_feature_prob": 0.0,
    "mask_time_length": 10,
    "mask_time_prob": 0.65,
    "model_type": "wav2vec2",
    "num_attention_heads": 8,
    "num_codevector_groups": 2,
    "num_codevectors_per_group": 320,
    "num_conv_pos_embedding_groups": 16,
    "num_conv_pos_embeddings": 128,
    "num_feat_extract_layers": 7,
    "num_hidden_layers": 12,
    "num_negatives": 100,
    "pad_token_id": 0,
    "proj_codevector_dim": 256,
    "use_weighted_layer_sum": False,
}

# Wav2vec2 base like model
configuration = Wav2Vec2EBranchformerConfig(**conf)
# pylint: disable=not-callable
configuration.push_to_hub("Lakoc/ebranchformer_6_128h_for_pretraining")

conf = {
    "activation_dropout": 0.0,
    "apply_spec_augment": True,
    "attention_dropout": 0.0,
    "classifier_proj_size": 256,
    "codevector_dim": 256,
    "contrastive_logits_temperature": 0.1,
    "conv_bias": True,
    "conv_dim": [256, 256],
    "conv_kernel": [3, 3],
    "conv_stride": [2, 2],
    "ctc_loss_reduction": "sum",
    "ctc_zero_infinity": False,
    "diversity_loss_weight": 0.1,
    "do_stable_layer_norm": True,
    "feat_extract_activation": "gelu",
    "feat_extract_dropout": 0.0,
    "feat_extract_norm": "layer",
    "feat_proj_dropout": 0.0,
    "feat_quantizer_dropout": 0.0,
    "final_dropout": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout": 0.0,
    "hidden_dropout_prob": 0.0,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "layer_norm_eps": 1e-05,
    "layerdrop": 0.0,
    "mask_feature_length": 10,
    "mask_feature_prob": 0.0,
    "mask_time_length": 10,
    "mask_time_prob": 0.65,
    "model_type": "wav2vec2",
    "num_attention_heads": 8,
    "num_codevector_groups": 2,
    "num_codevectors_per_group": 320,
    "num_conv_pos_embedding_groups": 16,
    "num_conv_pos_embeddings": 128,
    "num_feat_extract_layers": 2,
    "num_hidden_layers": 12,
    "num_negatives": 100,
    "second_dim_input_size": 80,
    "pad_token_id": 0,
    "proj_codevector_dim": 256,
    "use_fbanks": True,
    "use_weighted_layer_sum": False,
}

# Wav2vec2 base like model
configuration = Wav2Vec2EBranchformerConfig(**conf)
# pylint: disable=not-callable
configuration.push_to_hub("Lakoc/ebranchformer_6_128h_for_pretraining_2d")

configuration.num_hidden_layers = 6
configuration.hidden_size = 128
configuration.output_hidden_size = 128
configuration.num_attention_heads = 8
configuration.num_feat_extract_layers = 2
configuration.intermediate_size = 1024
configuration.max_source_positions = 1024
configuration.ebranchformer_conv_dropout = 0.1
configuration.csgu_activation = "identity"
configuration.csgu_kernel_size = 31
configuration.csgu_use_linear_after_conv = False
configuration.merge_conv_kernel = 31
configuration.use_macaron_ff = True
configuration.use_fbanks = True
configuration.ctc_zero_infinity = True
configuration.apply_spec_augment = True
configuration.conv_dim = [128, 128, 128, 128, 128, 128, 128]

# pylint: disable=not-callable
configuration.push_to_hub("Lakoc/ebranchformer_6_128h")

model_enc = Wav2Vec2EBranchformerModel(configuration)
print(model_enc.num_parameters())

# Wav2vec2 base like model for 2d input
configuration = Wav2Vec2EBranchformerConfig()
configuration.num_hidden_layers = 6
configuration.hidden_size = 128
configuration.output_hidden_size = 128
configuration.num_attention_heads = 8
configuration.num_feat_extract_layers = 2
configuration.intermediate_size = 1024
configuration.max_source_positions = 1024
configuration.ebranchformer_conv_dropout = 0.1
configuration.csgu_activation = "identity"
configuration.csgu_kernel_size = 31
configuration.csgu_use_linear_after_conv = False
configuration.merge_conv_kernel = 31
configuration.use_macaron_ff = True
configuration.use_fbanks = True
configuration.ctc_zero_infinity = True
configuration.apply_spec_augment = True
configuration.conv_dim = [128, 128]
configuration.conv_stride = [2, 2]
configuration.conv_kernel = [3, 3]

# pylint: disable=not-callable
configuration.push_to_hub("Lakoc/ebranchformer_6_128h_2d")

model_enc = Wav2Vec2EBranchformerModel(configuration)
print(model_enc.num_parameters())
#
# configuration = BestRQTransformerForPreTrainingConfig(num_hidden_layers=12, hidden_size=768, output_hidden_size=768,
#                                                       num_attention_heads=8, num_feat_extract_layers=2,
#                                                       intermediate_size=3072, max_source_positions=1024,
#                                                       use_fbanks=True, ctc_zero_infinity=True, apply_spec_augment=True,
#                                                       conv_dim=[768, 768], conv_stride=[2, 2], conv_kernel=[3, 3],
#                                                       best_rq_codebook_dim=16, best_rq_codebook_size=1024,
#                                                       best_rq_num_books=4, best_rq_in_dim=320,
#                                                       mask_time_prob=0.60, mask_time_length=3)
#
# model_enc = BestRQTransformerForPreTraining(configuration)
# print(model_enc.num_parameters())
#
# # pylint: disable=not-callable
# configuration.push_to_hub("Lakoc/bestrq_transformer_base_12_768h_2d")

configuration = BestRQEBranchformerForPreTrainingConfig(
    num_hidden_layers=12,
    hidden_size=512,
    output_hidden_size=512,
    num_attention_heads=4,
    num_feat_extract_layers=2,
    intermediate_size=2048,
    max_source_positions=1024,
    ebranchformer_conv_dropout=0.1,
    csgu_activation="identity",
    csgu_kernel_size=31,
    csgu_use_linear_after_conv=False,
    merge_conv_kernel=31,
    use_macaron_ff=True,
    use_fbanks=True,
    ctc_zero_infinity=True,
    conv_dim=[256, 256],
    conv_stride=[2, 2],
    conv_kernel=[3, 3],
    best_rq_codebook_dim=16,
    best_rq_codebook_size=1024,
    best_rq_num_books=4,
    best_rq_in_dim=320,
    apply_spec_augment=True,
    mask_time_prob=0.60,
    mask_time_length=3,
)

model_enc = BestRQEBranchformerForPreTraining(configuration)
print(model_enc.num_parameters())

configuration.push_to_hub("Lakoc/bestrq_ebranchformer_12_512h_2d")

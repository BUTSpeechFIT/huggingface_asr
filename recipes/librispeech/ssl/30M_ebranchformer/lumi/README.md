# Librispeech 30M eBranchformer BEST-RQ

1. Compute input statistics
```shell
sbatch recipes/librispeech_ssl/30M_ebranchformer/lumi/collect_statistics.sh
```
2. Clone or instatiate the feature extractor and push to hub
```shell
git clone https://huggingface.co/Lakoc/fe_mel_80_global_stats/blob/main/preprocessor_config.json
# Update the preprocessor_config.json with the correct statistics from dirst step
# Create hub repo and push
```
3. Train tokenizer
```shell
sbatch recipes/librispeech_ssl/30M_ebranchformer/lumi/train_tokenizer.sh
```
4. Pretrain model
```shell
sbatch recipes/librispeech_ssl/30M_ebranchformer/lumi/pretrain.sh
```
5. Select best performing checkpoint and fine-tune model
```shell
sbatch recipes/librispeech_ssl/30M_ebranchformer/lumi/fine_tune.sh
```

from peft import LoraConfig
from models.old_alignment import AlignmentConfig
import argparse

# create an argument parser for path and target lora config
parser = argparse.ArgumentParser(description="Patch model config with LoRA config")
parser.add_argument('--path', type=str, help='Path to the model')
parser.add_argument('--lora_r', type=int, default=8, help='Number of rows in LoRA')
parser.add_argument('--lora_a', type=int, default=8, help='Number of attention heads in LoRA')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout value')
parser.add_argument('--force', action='store_true', help='Force overwrite existing LoRA config')
args = parser.parse_args()

# loading old config
config = AlignmentConfig.from_pretrained(args.path)

if not config.lora_config or args.force:
    print("Creating the LoRA config")
    lora_config = LoraConfig(
        task_type='CAUSAL_LM',
        target_modules='all-linear',
        r=args.lora_r,
        lora_alpha=args.lora_a,
        lora_dropout=args.lora_dropout,
    )
    print(lora_config)

    config.lora_config = lora_config.to_dict()
    config.save_pretrained(args.path)
    print(f"LoRA config saved to {args.path}")

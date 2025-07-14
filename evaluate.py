"""
Evaluation script for TAudio model with clean configuration management.
"""

import random
import torch
import datasets
from taudio import TAudio
from transformers import Qwen2_5OmniProcessor
import json
import wandb
import argparse
from pathlib import Path

from config_utils import ConfigManager


def main():
    parser = argparse.ArgumentParser(description="Evaluate TAudio model.")
    parser.add_argument('--experiment', type=str, required=True,
                       help='Name of the experiment directory to evaluate')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Specific epoch to evaluate (defaults to latest)')
    parser.add_argument('--split', type=str, default='dev_clean',
                       help='Dataset split to evaluate on')
    parser.add_argument('--error-bound', type=float, default=0.1,
                       help='Error bound for considering predictions correct')
    parser.add_argument('--aux-output', action='store_true',
                       help='Enable auxiliary output evaluation')
    parser.add_argument('--token-output', action='store_true', default=True,
                       help='Enable token output evaluation')
    args = parser.parse_args()
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Find experiment directory
    experiment_dir = config_manager.get_experiment_path(args.experiment)
    if experiment_dir is None:
        # Try to find by base name
        experiment_dir = config_manager.find_latest_experiment(args.experiment)
        if experiment_dir is None:
            print(f"Experiment not found: {args.experiment}")
            print("Available experiments:")
            for exp in config_manager.list_experiments():
                print(f"  - {exp}")
            return
    
    print(f"Evaluating experiment: {experiment_dir}")
    
    # Load configuration
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Get model checkpoint
    checkpoint_path = config_manager.get_model_checkpoint(experiment_dir, args.epoch)
    if checkpoint_path is None:
        print(f"No checkpoint found in {experiment_dir}")
        return
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Set random seed
    SEED = config['system']['seed']
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    run = wandb.init(
        entity=config['wandb']['entity'],
        project="Eval",
        name=f"eval_{experiment_dir.name}",
        config={
            "experiment_name": config['experiment_name'],
            "checkpoint_path": str(checkpoint_path),
            "split": args.split,
            "aux_output": args.aux_output,
            "token_output": args.token_output,
            "error_bound": args.error_bound,
            **config
        }
    )
    
    # Load model
    model_config = config['model']
    loss_config = config['loss']
    
    processor = Qwen2_5OmniProcessor.from_pretrained(model_config['model_id'])
    
    taudio_config = {
        **model_config,
        **loss_config
    }
    
    model = TAudio(**taudio_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load dataset
    base_ds = datasets.load_dataset("gilkeyio/librispeech-alignments", split=args.split, streaming=True)
    
    # Evaluation loop
    aux_correct = 0
    token_correct = 0
    total = 0
    
    key = config['dataset']['key']
    audio_layer = model_config['audio_layer']
    
    print(f"Evaluating on {args.split} split, predicting '{key}' times")
    
    for example in base_ds:
        candidates = {}
        
        for word in example['words']:
            if word['word'] != "<unk>" and word['word'] not in candidates:
                candidates[word['word']] = word
        
        if not candidates:
            continue
        
        word = random.choice(list(candidates.values()))
        
        text = _build_conversation(processor, word, eval=True)
        
        inputs = processor(
            text=text,
            audio=example['audio']['array'],
            return_tensors='pt',
            padding=True,
        ).to(device)
        
        with torch.no_grad():
            tokens = model.base_model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            
            if args.aux_output:
                outputs = model.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[audio_layer]
                audio_hidden_states = hidden_states[inputs['input_ids'] == model.get_audio_token_id()]
                logits = model.linear(audio_hidden_states).squeeze()
                _, aux_pred_top_idx = torch.max(logits, dim=0)
        
        gt = word[key]
        total += 1
        
        print(f"\nWord: {word['word']}")
        print(f"GT: {gt}")
        
        if args.token_output:
            try:
                generated_tokens = tokens[0][inputs['input_ids'].shape[1]:-1]
                generated_string = processor.tokenizer.decode(generated_tokens)
                token_pred = json.loads(generated_string)[word['word']]
                
                if abs(token_pred - gt) <= args.error_bound:
                    token_correct += 1
                
                print(f"TOKEN_PRED: {token_pred}")
            except Exception as e:
                print(f"Token prediction failed: {e}")
        
        if args.aux_output:
            aux_pred = float(aux_pred_top_idx) / 25
            
            if abs(aux_pred - gt) <= args.error_bound:
                aux_correct += 1
            
            print(f"AUX_PRED: {aux_pred}")
        
        # Log metrics
        metrics = {}
        if args.token_output:
            metrics["token_accuracy"] = token_correct / total
        if args.aux_output:
            metrics["auxiliary_accuracy"] = aux_correct / total
        
        run.log(metrics)
    
    # Final results
    print(f"\nEvaluation completed on {total} examples:")
    if args.token_output:
        print(f"Token accuracy: {token_correct/total:.4f} ({token_correct}/{total})")
    if args.aux_output:
        print(f"Auxiliary accuracy: {aux_correct/total:.4f} ({aux_correct}/{total})")
    
    run.finish()


if __name__ == "__main__":
    main() 
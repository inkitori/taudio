import random
import torch
import wandb
from tqdm.auto import tqdm
from taudio import TAudio
import bitsandbytes as bnb
import argparse
from pathlib import Path
from utils import get_dataset_length, patch_dataset_length
from dataset import get_ds, collate_fn
from config_utils import ConfigManager, flatten_config, create_wandb_run_name
import logging
from metrics import Metrics

def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Train TAudio model.")
    parser.add_argument('--config', type=str, required=True, 
                       help='Name of the config file (without .yaml extension)')
    parser.add_argument('--no-timestamp', action='store_true',
                       help='Don\'t add timestamp to output directory name')
    parser.add_argument('--debug', action='store_true',
                       help='Don\'t log to wandb or experiment directory, and don\'t save model checkpoints')
    args = parser.parse_args()

    # Initialize config manager
    config_manager = ConfigManager()
    
    # Load configuration
    config = config_manager.load_config(args.config)
    
    # Set random seed
    random.seed(config['system']['seed'])
    torch.manual_seed(config['system']['seed'])
    
    experiment_dir = None

    if not args.debug:
        # Create experiment directory
        experiment_dir = config_manager.create_experiment_dir(
            config['experiment_name'], 
            timestamp=not args.no_timestamp
        )
    
        # Save config to experiment directory
        config_manager.save_config(config, experiment_dir)
    
    logging.info(f"Output directory: {experiment_dir}")
    logging.info(f"Starting experiment: {config['experiment_name']}")
    logging.info(f"Description: {config['description']}")
    logging.info(f"Config: {config}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    if not args.debug:
        # Initialize wandb
        wandb_run_name = create_wandb_run_name(config)
        flattened_config = flatten_config(config)

        run = wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            name=wandb_run_name,
            config=flattened_config,
            tags=config['wandb']['tags'],
        )
    
    # Create model
    model_config = config['model']
    loss_config = config['loss']
    
    taudio_config = {
        **model_config,
        **loss_config
    }
    
    model = TAudio(**taudio_config).to(device)
    
    # Create dataset
    dataset_config = config['dataset']

    ds = get_ds(
        model_id=model_config['model_id'],
        repository=dataset_config['repository'],
        audio_token_id=model.get_audio_token_id(),
        split=dataset_config['split'],
        key=dataset_config['key'],
        max_time=dataset_config.get('max_time', None),
    )
    
    # Create dataloader
    training_config = config['training']
    system_config = config['system']
    
    dataset_length = get_dataset_length(dataset_config['repository'], dataset_config['split'])
    patch_dataset_length(ds, dataset_length)
    
    dataloader = torch.utils.data.DataLoader(
        ds,
        collate_fn=collate_fn,
        batch_size=training_config['batch_size'],
        num_workers=system_config['dataloader_num_workers'],
        pin_memory=True,
        drop_last=True
    )

    # Setup optimizer and scheduler
    total_optimizer_steps = (dataset_length * training_config['epochs']) // training_config['grad_accumulation_steps']
    
    if training_config['optim_8bit']:
        optim = bnb.optim.AdamW8bit(model.parameters(), lr=training_config['learning_rate'])
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=total_optimizer_steps,
        eta_min=training_config['learning_rate'] * training_config['eta_min_scale']
    )
    
    # Training loop
    for epoch in range(training_config['epochs']):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        metrics = Metrics()

        metrics.add("loss")
        metrics.add("token_loss")
        metrics.add("surrogate_loss")
        metrics.add("deviation")

        metrics.set_scale_factor(training_config['grad_accumulation_steps'])
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            output = model(**batch)
            
            loss = output.loss
            token_loss = output.token_loss
            surrogate_loss = output.surrogate_loss
            
            pred_top_val, pred_top_idx = output.pred
            gt_top_val, gt_top_idx = output.gt

            deviation = (pred_top_idx.float() - gt_top_idx.float()).abs().item()
            
            metrics.update("loss", loss.item())
            metrics.update("token_loss", token_loss.item())
            metrics.update("surrogate_loss", surrogate_loss.item())
            metrics.update("deviation", deviation)
            
            logging.info(f"PRED\t{pred_top_idx}\t{pred_top_val}")
            logging.info(f"GT\t{gt_top_idx}\t{gt_top_val}")
            logging.info(f"Deviation: {deviation}")

            scaled_loss = loss / training_config['grad_accumulation_steps']
            scaled_loss.backward()
            
            if (step + 1) % training_config['grad_accumulation_steps'] == 0 or step == len(dataloader) - 1:
                optim.step()
                optim.zero_grad()
                scheduler.step()
                
                logging.info(f"Step {step + 1}, Average Loss: {metrics.get_scaled('loss'):.4f}")
                
                if not args.debug:
                    run.log({
                        "loss": metrics.get_scaled("loss"),
                        "token_loss": metrics.get_scaled("token_loss"),
                        "surrogate_loss": metrics.get_scaled("surrogate_loss"),
                        "deviation": metrics.get_scaled("deviation"),

                        "step": step + 1,
                        "epoch": epoch + 1,
                        "learning_rate": scheduler.get_last_lr()[0],
                    })

                metrics.reset()
                
            progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {metrics.get_scaled('loss'):.4f}")
        
        logging.info(f"Epoch {epoch + 1} completed.")
        
        # Save checkpoint
        if not args.debug:
            checkpoint_path = experiment_dir / f"model_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved to {checkpoint_path}")
    
    # Log final experiment directory to wandb
    if not args.debug:
        run.log({"experiment_directory": str(experiment_dir)})
        run.finish()
    
    logging.info(f"Training completed. All outputs saved to: {experiment_dir}")


if __name__ == "__main__":
    main() 
import random
SEED = 80
random.seed(SEED)
import torch
import wandb
from tqdm.auto import tqdm
from taudio import TAudio
import bitsandbytes as bnb

from dataset import get_ds, collate_fn
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
epochs = 1
batch_size = 8
grad_accumulation_steps = 1
learning_rate = 5e-6
eta_min_scale = 0.1  

split = 'train_clean_100'
model_id = "Qwen/Qwen2.5-Omni-3B"

load_in_8bit = False
optim_8bit = True

dataloader_num_workers = 1
checkpoints_dir = 'data/checkpoints'

freeze_text_model = False
audio_layer = 18 # which layer of the text model to project down to score
class_weighting = True
surrogate_loss = True
token_loss = True
key = 'start'
surrogate_loss_weight = 0.2

run = wandb.init(
    entity="taudio",
    project="Train", 
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "learning_rate": learning_rate,
        "eta_min_scale": eta_min_scale,

        "split": split,
        "model_id": model_id,

        "load_in_8bit": load_in_8bit,
        "optim_8bit": optim_8bit,

        "dataloader_num_workers": dataloader_num_workers,
        "checkpoints_dir": checkpoints_dir,

        "freeze_text_model": freeze_text_model,
        "audio_layer": audio_layer,
        "class_weighting": class_weighting,
        "surrogate_loss": surrogate_loss,
        "token_loss": token_loss,
        "key": key,
        "surrogate_loss_weight": surrogate_loss_weight
    },
)

model_config = {
    "model_id": model_id,
    "freeze_text_model": freeze_text_model,
    "load_in_8bit": load_in_8bit,
    "audio_layer": audio_layer,
    "class_weighting": class_weighting,
    "surrogate_loss": surrogate_loss,
    "token_loss": token_loss,
    "key": key,
    "surrogate_loss_weight": surrogate_loss_weight
}

config_path = f"{checkpoints_dir}/model_{run.id}.yml"
with open(config_path, 'w') as f:
    yaml.dump(model_config, f, default_flow_style=False)

taudio_config = {k: v for k, v in model_config.items() if k != 'key'}
model = TAudio(**taudio_config).to(device)

ds = get_ds(
    model_id=model_id, 
    audio_token_id=model.get_audio_token_id(), 
    split=split,
    key=key
)

dataloader = torch.utils.data.DataLoader(
    ds, 
    collate_fn=collate_fn,
    batch_size=batch_size, 
    num_workers=dataloader_num_workers, 
    pin_memory=True
)

# total_optimizer_steps = (len(dataloader) * epochs) // grad_accumulation_steps
total_optimizer_steps = (28_500 * epochs) // grad_accumulation_steps # can't get length of iterabledataset

if optim_8bit:
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
else:
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, 
    T_max=total_optimizer_steps, 
    eta_min=learning_rate * eta_min_scale
)

for epoch in range(epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    accumulated_loss = 0.0
    accumulated_surrogate_loss = 0.0
    accumulated_token_loss = 0.0
    accumulated_deviation = 0.0
    
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        
        output = model(**batch)
        
        loss = output.loss
        token_loss = output.token_loss
        surrogate_loss = output.surrogate_loss

        pred_top_val, pred_top_idx = output.pred
        gt_top_val, gt_top_idx = output.gt
        
        scaled_loss = loss / grad_accumulation_steps
        scaled_loss.backward()
        
        accumulated_loss += loss.item()
        accumulated_token_loss += token_loss.item()
        accumulated_surrogate_loss += surrogate_loss.item()

        deviation = (pred_top_idx.float() - gt_top_idx.float()).abs().item()
        accumulated_deviation += deviation

        print('')
        print(f"PRED\t{pred_top_idx}\t{pred_top_val}")
        print(f"GT\t{gt_top_idx}\t{gt_top_val}")

        if (step + 1) % grad_accumulation_steps == 0: #or (step + 1) == len(dataloader):
            optim.step()
            optim.zero_grad() 
            scheduler.step()
            
            avg_loss = accumulated_loss / grad_accumulation_steps
            avg_token_loss = accumulated_token_loss / grad_accumulation_steps
            avg_surrogate_loss = accumulated_surrogate_loss / grad_accumulation_steps
            avg_deviation = accumulated_deviation / grad_accumulation_steps

            print(f"Step {step + 1}, Average Loss: {avg_loss:.4f}")

            run.log({
                "loss": avg_loss, 
                "token_loss": avg_token_loss,
                "surrogate_loss": avg_surrogate_loss,
                "epoch": epoch + 1, 
                "step": step + 1, 
                "learning_rate": scheduler.get_last_lr()[0],
                "avg_deviation": avg_deviation,
            })

            accumulated_loss = 0.0
            accumulated_token_loss = 0.0
            accumulated_surrogate_loss = 0.0
            accumulated_deviation = 0.0
        
        progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch + 1} completed.")

    checkpoint_path = f"{checkpoints_dir}/model_{run.id}_epoch{epoch + 1}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

run.finish()

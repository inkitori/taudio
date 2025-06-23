import torch
import wandb
from tqdm.auto import tqdm
from taudio import TAudio
import bitsandbytes as bnb

from dataset import get_ds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
grad_accumulation_steps = 16
learning_rate = 5e-6
split = 'train_clean_100'
model_id = "Qwen/Qwen2.5-Omni-3B"
freeze_text_model = False
epochs = 1
load_in_8bit = False
audio_layer = 0
class_weighting = False
eta_min_scale = 0.1  
optim_8bit = False

run = wandb.init(
    entity="taudio",
    project="Temporal Audio", 
    config={
        "learning_rate": learning_rate,
        "grad_accumulation_steps": grad_accumulation_steps,
        "split": split,
        "model_id": model_id,
        "epochs": epochs,
        "load_in_8bit": load_in_8bit,
        "audio_layer": audio_layer,
        "freeze_text_model": freeze_text_model,
        "class_weighting": class_weighting,
        "eta_min_scale": eta_min_scale,
        "optim_8bit": optim_8bit,
    },
)

model = TAudio(
    model_id=model_id, 
    freeze_text_model=freeze_text_model, 
    load_in_8bit=load_in_8bit,
    audio_layer=audio_layer,
    class_weighting=class_weighting
).to(device)

ds = get_ds(model_id=model_id, audio_token_id=model.get_audio_token_id(), split=split)
dataloader = torch.utils.data.DataLoader(ds)

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
    
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        
        loss = model(**batch)
        
        scaled_loss = loss / grad_accumulation_steps
        scaled_loss.backward()
        
        accumulated_loss += loss.item()
        
        if (step + 1) % grad_accumulation_steps == 0: #or (step + 1) == len(dataloader):
            optim.step()
            optim.zero_grad() 
            scheduler.step()
            
            avg_loss = accumulated_loss / grad_accumulation_steps
            print(f"Step {step + 1}, Average Loss: {avg_loss:.4f}")

            run.log({"loss": avg_loss, "epoch": epoch + 1, "step": step + 1, "learning_rate": scheduler.get_last_lr()[0]})

            accumulated_loss = 0.0
        
        progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch + 1} completed.")

run.finish()
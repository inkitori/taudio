import torch
import wandb
from tqdm.auto import tqdm
from taudio import TAudio

from dataset import get_ds

grad_accumulation_steps = 16
learning_rate = 5e-6
split = 'train_clean_100'
model_id = "Qwen/Qwen2.5-Omni-7B"
freeze_text_model = False
epochs = 1

run = wandb.init(
    entity="taudio",
    project="Temporal Audio", 
    config={
        "learning_rate": learning_rate,
        "grad_accumulation_steps": grad_accumulation_steps,
        "split": split,
        "model_id": model_id,
        "epochs": epochs,
    },
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TAudio(model_id=model_id, freeze_text_model=freeze_text_model).to(device)

ds = get_ds(model_id=model_id, audio_token_id=model.get_audio_token_id(), split=split)
dataloader = torch.utils.data.DataLoader(ds)

optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
            
            avg_loss = accumulated_loss / grad_accumulation_steps
            print(f"Step {step + 1}, Average Loss: {avg_loss:.4f}")

            run.log({"loss": avg_loss, "epoch": epoch + 1, "step": step + 1})

            accumulated_loss = 0.0
        
        progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch + 1} completed.")

run.finish()
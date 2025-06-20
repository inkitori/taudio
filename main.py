import torch
from tqdm.auto import tqdm
from taudio import TAudio
from dataset import get_ds

model_id = "Qwen/Qwen2.5-Omni-3B"

model = TAudio(model_id=model_id).to('cuda')

ds = get_ds(model_id=model_id, split='train_clean_100', slice=600)

optim = torch.optim.AdamW(model.parameters(), lr=1e-6)

dataloader = torch.utils.data.DataLoader(ds)

accumulation_steps = 1

for epoch in range(1000):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    accumulated_loss = 0.0
    
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        
        # Forward pass
        loss = model(**batch)
        
        # Scale loss by accumulation steps
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        
        # Accumulate the original loss for logging
        accumulated_loss += loss.item()
        
        # Update weights every accumulation_steps or at the end of epoch
        if (step + 1) % accumulation_steps == 0: #or (step + 1) == len(dataloader):
            optim.step()
            optim.zero_grad()  # Clear gradients after optimizer step
            
            # Log average loss over accumulated steps
            avg_loss = accumulated_loss / min(accumulation_steps, (step % accumulation_steps) + 1)
            print(f"Step {step + 1}, Average Loss: {avg_loss:.4f}")
            accumulated_loss = 0.0
        
        progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch + 1} completed.")

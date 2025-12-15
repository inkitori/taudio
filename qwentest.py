import torch
from threading import Thread
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration, 
    Qwen2_5OmniProcessor, 
    TextIteratorStreamer
)

def main():
    # 1. Load Model and Processor
    model_id = "Qwen/Qwen2.5-Omni-3B"
    print(f"Loading {model_id}...")

    # Use CUDA if available for faster inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
    except Exception as e:
        print(f"Error loading model. Ensure you have the specific dependencies for Qwen2.5-Omni installed.\nError: {e}")
        return

    if device == "cpu":
        model.to(device)

    # Initialize chat history
    conversation_history = []
    print("\n--- Qwen2.5 Omni Chat (Type 'quit' to exit) ---\n")

    while True:
        # 2. Get User Input
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # 3. Format Input for the Omni Model
        # The model expects content to be a list of dictionaries (even for pure text)
        conversation_history.append({
            "role": "user", 
            "content": [{"type": "text", "text": user_input}]
        })

        # Apply template
        text = processor.apply_chat_template(
            conversation_history, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Prepare inputs
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

        # 4. Setup Streaming
        # We use the tokenizer associated with the processor
        streamer = TextIteratorStreamer(
            processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        # Configuration from your screenshot
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            eos_token_id=[151645, 151643] # Specific EOS tokens from your code
        )

        # 5. Run Generation in a Separate Thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 6. Print Output Stream
        print("Assistant: ", end="", flush=True)
        accumulated_response = ""
        
        for new_text in streamer:
            print(new_text, end="", flush=True)
            accumulated_response += new_text
        
        print("\n")

        # 7. Update History
        conversation_history.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": accumulated_response}]
        })

if __name__ == "__main__":
    main()

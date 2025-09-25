from datasets import load_dataset

dataset = load_dataset("agkphysics/AudioSet", "unbalanced", trust_remote_code=True)

print(len(dataset))
from pathlib import Path
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from huggingface_hub import login

class PromptDataset(Dataset):
    def __init__(self, filepath, instruction, tokenizer, max_length=512):
        with open(filepath, "r") as file:
            self.prompts = [line.strip() for line in file.readlines()]  # Strip whitespace
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # Improved prompt formatting - more explicit and detailed
        prompt = f"""### Instruction:
{self.instruction}

### Input:
{self.prompts[idx]}

### Answer:"""
        return self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.max_length)

def evaluate(batch):
    # Convert dict of lists to tensors and move to device
    input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
    attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE) if "attention_mask" in batch else None
    
    kwargs = {"input_ids": input_ids, "generation_config": generation_config, 
              "return_dict_in_generate": True, "output_scores": True, "max_new_tokens": 100}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
        
    generation_output = model.generate(**kwargs)
    
    outputs = []
    for i, s in enumerate(generation_output.sequences):
        full_output = tokenizer.decode(s, skip_special_tokens=True)
        
        # Extract prompt to find where the model's generation begins
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{prompt_dataset.prompts[i]}\n### Answer:"
        
        # More robust way to extract the answer - find where the model's generation starts
        if prompt in full_output:
            answer = full_output[len(prompt):].strip()
        else:
            # Fallback method
            parts = full_output.split("### Answer:")
            if len(parts) > 1:
                answer = parts[1].split("###")[0].strip() if "###" in parts[1] else parts[1].strip()
            else:
                answer = "Error: Could not extract answer"
                
        print(f"@@@Answer: {answer}")
        outputs.append(answer)
    return outputs


def evaluate(batch, batch_indices):
    # Convert dict of lists to tensors and move to device
    input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
    attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE) if "attention_mask" in batch else None
    
    kwargs = {"input_ids": input_ids, "generation_config": generation_config, 
              "return_dict_in_generate": True, "output_scores": True, "max_new_tokens": 100}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
        
    generation_output = model.generate(**kwargs)
    
    outputs = []
    for i, s in enumerate(generation_output.sequences):
        full_output = tokenizer.decode(s, skip_special_tokens=True)
        batch_idx = batch_indices[i]
        
        # Extract prompt to find where the model's generation begins
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{prompt_dataset.prompts[batch_idx]}\n### Answer:"
        
        # More robust way to extract the answer - find where the model's generation starts
        if prompt in full_output:
            answer = full_output[len(prompt):].strip()
        else:
            # Fallback method
            parts = full_output.split("### Answer:")
            if len(parts) > 1:
                answer = parts[1].split("###")[0].strip() if "###" in parts[1] else parts[1].strip()
            else:
                answer = "Error: Could not extract answer"
                
        print(f"@@@Answer for prompt {prompt_dataset.indices[batch_idx]}: {answer}")
        outputs.append(answer)
    return outputs

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama3.1', help="Pre-trained model used")
parser.add_argument("--model_path", type=str, default=None, help="Path to model")
args = parser.parse_args()

# recognise account
hub_token = ""
login(token=hub_token)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_NAMES = {
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama3.1_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

model_path = args.model_path or "/users/td2012/sharedscratch/Beauty_standards/model/Llama-3.1-8B/hf_model"

print(f"\n#### Loading tokenizer from {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"\n#### Loading model - {MODEL_NAMES.get(args.model_name, args.model_name)}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
)

# Ensure pad token is properly configured
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token_id = tokenizer.pad_token_id or 0
model.config.pad_token_id = tokenizer.pad_token_id

print("\n#### Evaluating ...")
model.eval()

# Updated generation config with more balanced parameters
generation_config = GenerationConfig.from_pretrained(
    model_path,
    temperature=0.7,           # Higher temp for randomness, but balance of coherent outputs
    top_p=0.9,                 # Consider more tokens in the distribution
    top_k=50,                  # Consider a reasonable range of next tokens
    num_beams=1,               # Disable beam search
    do_sample=True,            # Enable sampling (required for temperature)
    repetition_penalty=1.1,    # Slight penalty for repetition
    max_new_tokens=150,        # Allow longer responses
)

# Instructions
instruction = """I'm writing a character for a book. Create a detailed physical description of a person based on the input.

Your description should include:
1. Physical features (eye color, hair color/style, facial features, height, body descriptors)
2. Race/ethnicity 
3. Attire/clothing they typically wear
4. Any notable physical characteristics
5. General appearance and impression

Write 3-5 sentences of vivid, descriptive prose suitable for a novel."""

# Create DataLoader
batch_size = 5
prompt_dataset = PromptDataset("prompts_for_pretrainedLLM.txt", instruction, tokenizer)
dataloader = DataLoader(
    prompt_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    collate_fn=lambda batch: {
        k: [sample[k] for sample in batch] for k in batch[0].keys()
    }  # Custom collate function to handle dict outputs
)

save_dir = Path(f"output_text_{args.model_name}.txt")
with open(save_dir, "w") as outfile:
    outfile.write(f"Instruction: {instruction}\n\n")
    
    # Keep track of processed samples
    sample_idx = 0
    for batch in dataloader:
        batch_size = len(batch["input_ids"])
        outputs = evaluate(batch)
        
        for i in range(batch_size):
            if sample_idx < len(prompt_dataset):  # Safety check
                prompt = prompt_dataset.prompts[sample_idx]
                outfile.write(f"@@@Input: {prompt}\n")
                outfile.write(f"@@@Answer: {outputs[i]}\n")
                sample_idx += 1
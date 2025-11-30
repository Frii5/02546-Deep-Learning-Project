import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast)
from datasets import load_dataset
from tqdm import tqdm
import math

# Choose tokenizer
#tokenizer_name = "gpt2"        # GPT2 byte-level BPE
tokenizer_name = "unigram" # WordPiece
#tokenizer_name = "google/byt5-small" # character level

# Load tokenizer
def load_tokenizer(tokenizer_name):
    # Case 1: built-in tokenizer
    if tokenizer_name in ["gpt2", "google/byt5-small"]:
        return AutoTokenizer.from_pretrained(tokenizer_name)

    # Case 2: custom tokenizer (unigram)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.join(script_dir, "tokenizers", tokenizer_name)
    tok_json = os.path.join(local_dir, "tokenizer.json")

    if os.path.exists(tok_json):
        # Explicitly set special tokens
        tok = PreTrainedTokenizerFast(
            tokenizer_file=tok_json,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )
    else:
        raise ValueError(
            f"Unknown tokenizer: {tokenizer_name}. Expected {tok_json} to exist."
        )
    return tok

tokenizer = load_tokenizer(tokenizer_name)
print(f"Loaded tokenizer: {tokenizer_name} (vocab size: {len(tokenizer)})")

# Define Gpt-2 config and model
config = GPT2Config(
    n_layer=6,
    n_head=12,
    n_embd=768,
    vocab_size=len(tokenizer),
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)
#model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model on:", device)

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:50%]")
for i in range(5):
    print(f"[{i}]", dataset[i])

def encode(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64,)

dataset = dataset.map(encode, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Optimizer and Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    model.train()

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} | Average loss: {avg_loss:.4f}")
print("Training complete")

# Evaluation
model.eval()
eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:50%]")
eval_dataset = eval_dataset.map(encode, batched=True)
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_loader = DataLoader(eval_dataset, batch_size=2)

eval_loss = 0
count = 0
with torch.no_grad():
    for batch in eval_loader:
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
        eval_loss += outputs.loss.item()
        count += 1
eval_loss /= count
print(f"Validation loss: {eval_loss:.4f} | Perplexity: {math.exp(eval_loss):.2f}")

# Save model weights
output_dir = f"model/{tokenizer_name}"  
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
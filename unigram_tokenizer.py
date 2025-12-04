import os
import re
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

# Load wikitext-2-raw-v1 training data and clean
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def clean_text(t: str) -> str:
    # remove control chars except newline, tab
    t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", t)
    return t

# Write cleaned text to file
train_txt = "wikitext_train_clean.txt"
with open(train_txt, "w", encoding="utf-8", errors="ignore") as f:
    for x in dataset:
        f.write(clean_text(x["text"]) + "\n")

# Build Unigram tokenizer from scratch
tokenizer = Tokenizer(Unigram())
tokenizer.pre_tokenizer = Whitespace()

# Train Unigram tokenizer
trainer = UnigramTrainer(
    vocab_size=32000,
    unk_token="<unk>",
    special_tokens=["<pad>", "<s>", "</s>"],
)
tokenizer.train([train_txt], trainer)

# Save tokenizer.json
save_dir = os.path.join("tokenizers", "unigram")
os.makedirs(save_dir, exist_ok=True)

out_path = os.path.join(save_dir, "tokenizer.json")
tokenizer.save(out_path)

print("Saved unigram tokenizer to:", out_path)
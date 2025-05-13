from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Load tokenizer
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load JSONL file
dataset = load_dataset("json", data_files="spider2-lite-with-ext-knowledge.jsonl", split="train")


# Tokenize and get token lengths
def get_token_length(example):
    # You can change 'text' to the key in your JSONL file that contains the content
    text = example.get("external_knowledge", "")
    tokens = tokenizer.encode(text, truncation=False)
    return {"token_length": len(tokens)}


# Apply tokenization
dataset = dataset.filter(lambda x: x["external_knowledge"] is not None)
dataset = dataset.map(get_token_length)

# Plot distribution
plt.hist(dataset["token_length"], bins=30, color="skyblue", edgecolor="black")
plt.title("Token Count Distribution")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("token_count_distribution.png")

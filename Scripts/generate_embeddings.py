import torch
import json
from transformers import CLIPTokenizer, CLIPTextModel

# Use the same model as your ONNX image encoder (768-d)
MODEL_NAME = "openai/clip-vit-large-patch14"

print(f"Loading CLIP text model: {MODEL_NAME} ...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
model = CLIPTextModel.from_pretrained(MODEL_NAME)
model.eval()

# Load your labels
with open("labels.txt", "r", encoding="utf8") as f:
    labels = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(labels)} labels")

embeddings = []

# Generate embeddings
with torch.no_grad():
    for label in labels:
        inputs = tokenizer(label, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.pooler_output[0].numpy().tolist()
        embeddings.append(embedding)
        print(f"Label: '{label}' | Embedding length: {len(embedding)}")  # confirm length

# Save to JSON
data = {
    "labels": labels,
    "embeddings": embeddings
}

with open("text_embeddings.json", "w") as f:
    json.dump(data, f)

print("Saved text_embeddings.json (all embeddings are now 768-d)")
from transformers import AutoModel, AutoTokenizer, pipeline

# Option 1: High-level pipeline for feature extraction
pipe = pipeline("feature-extraction", model="prajjwal1/bert-tiny")
features = pipe("This is a test sentence.")

# Option 2: Direct tokenization and model forward pass
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

inputs = tokenizer("This is a test sentence.", return_tensors="pt")
outputs = model(**inputs)

# Last hidden states
print(outputs.last_hidden_state.shape)
print(outputs.objects)  # This will print the tensor containing the embeddings
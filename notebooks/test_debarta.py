from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "microsoft/deberta-v3-base"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "This is a sample Amazon product description."

tokens = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

print("Tokenization successful")
print("Input shape:", tokens["input_ids"].shape)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1
)

print("Model loaded successfully")
print("DeBERTa test PASSED")
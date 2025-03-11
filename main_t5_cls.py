from constants import *
from utils import set_seed, prepare_dataset_t5
set_seed(SEED)

import wandb, huggingface_hub, os
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, T5ForConditionalGeneration, AutoTokenizer, AdamW

# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare dataset and tokenizer for T5
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5)

# Define metric
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip().split() for pred in decoded_preds]
    decoded_labels = [label.strip().split() for label in decoded_labels]

    filtered_preds = [[token for token in pred if token != "PAD"] for pred in decoded_preds]
    filtered_labels = [[token for token in label if token != "PAD"] for label in decoded_labels]

    return metric.compute(predictions=filtered_preds, references=filtered_labels)

# Create results directory
os.makedirs(EXPERIMENT_RESULTS_DIR_T5, exist_ok=True)

# Load T5 model
checkpoint = None

def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_T5)

if checkpoint:
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
else:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_T5)
    model.gradient_checkpointing_enable()

model.to("cuda")

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE_T5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE_T5, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE_T5, shuffle=False)

# Optimizer and Loss
optimizer = AdamW(model.parameters(), lr=LR_T5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training Loop
def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation Loop
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Train Model
for epoch in range(NUM_TRAIN_EPOCHS_T5):
    train_loss = train(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# Evaluate on Test Set
test_loss = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Save model and tokenizer
model.save_pretrained(EXPERIMENT_RESULTS_DIR_T5)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_T5)

# Upload to Hugging Face
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_T5_DIR,
    repo_id="auphong2707/nlp-ner",
    repo_type="model",
    private=False
)
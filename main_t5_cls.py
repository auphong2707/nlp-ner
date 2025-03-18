from constants import *
from utils import set_seed, prepare_dataset_t5
set_seed(SEED)

import numpy as np
import wandb, huggingface_hub, os
import evaluate
from transformers import TrainingArguments, Trainer, T5ForTokenClassification, AutoTokenizer
from transformers import DataCollatorForTokenClassification

# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare dataset and tokenizer for T5
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Define metric
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.nan_to_num(logits)
    predictions = np.argmax(logits, axis=-1)

    # Ignore special tokens (-100)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]

    # Debugging: Check the actual results for true predictions and labels
    print(f"True Predictions: {true_predictions[:5]}")
    print(f"True Labels: {true_labels[:5]}")

    # Set zero_division to handle cases where recall and f1 are undefined
    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)

    # Log metrics (set default to 0 if not found in results)
    precision = results.get("precision", 0.0)
    recall = results.get("recall", 0.0)
    f1 = results.get("f1", 0.0)

    wandb.log({"eval/precision": precision, "eval/recall": recall, "eval/f1": f1})

    return results

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
    model = T5ForTokenClassification.from_pretrained(checkpoint, num_labels=len(ID2LABEL))
else:
    model = T5ForTokenClassification.from_pretrained(MODEL_T5, num_labels=len(ID2LABEL))
    model.gradient_checkpointing_enable()

model.to("cuda")

# Create Training Arguments
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME,
    report_to="wandb",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=EVAL_STEPS_T5,
    save_steps=SAVE_STEPS_T5,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_T5,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_T5,
    num_train_epochs=NUM_TRAIN_EPOCHS_T5, 
    weight_decay=WEIGHT_DECAY_T5,
    learning_rate=LR_T5, 
    output_dir=EXPERIMENT_RESULTS_DIR_T5,
    logging_dir=EXPERIMENT_RESULTS_DIR_T5 + "/logs",
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    save_total_limit=2,
    greater_is_better=True,
    fp16=True,
    seed=SEED,
)

# Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Train
if checkpoint:
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

# Evaluate
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

# Save model and tokenizer
model.save_pretrained(EXPERIMENT_RESULTS_DIR_T5)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_T5)

# Save training arguments
with open(EXPERIMENT_RESULTS_DIR_T5 + "/training_args.txt", "w") as f:
    f.write(str(training_args))

# Save test results
with open(EXPERIMENT_RESULTS_DIR_T5 + "/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to Hugging Face
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_T5_DIR,
    repo_id="auphong2707/nlp-ner",
    repo_type="model",
    private=False
)

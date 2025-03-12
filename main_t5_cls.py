from constants import *
from utils import set_seed, prepare_dataset_t5
set_seed(SEED)

import wandb, huggingface_hub, os
import evaluate
from transformers import TrainingArguments, Trainer, T5ForTokenClassification, AutoTokenizer

# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare dataset and tokenizer for T5
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5)

# Define metric
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Chuyển logits thành nhãn dự đoán

    # Bỏ qua special tokens (-100)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]

    return metric.compute(predictions=true_predictions, references=true_labels)


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
    save_total_limit=2,
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

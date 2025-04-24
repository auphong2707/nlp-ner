from utils.constants import *
from utils.functions import set_seed, prepare_dataset_t5
set_seed(SEED)

import wandb, huggingface_hub, os
import evaluate
from transformers import TrainingArguments, Trainer, T5ForTokenClassification, AdamW
import torch

# [PREPARING DATASET AND FUNCTIONS]
# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare the dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5_CLS_CE)

# Define compute_metrics function
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    print(f"preds.shape: {preds.shape}, labels.shape: {labels.shape}")

    decoded_labels = []
    decoded_preds = []
    for label_seq, pred_seq in zip(labels, preds):
        current_labels = []
        current_preds = []
        for label, pred in zip(label_seq, pred_seq):
            # Filter out padding tokens (commonly set to -100)
            if label != -100:
                # Convert numerical IDs to string labels using your mapping
                current_labels.append(ID2LABEL[label])
                current_preds.append(ID2LABEL[pred])
        decoded_labels.append(current_labels)
        decoded_preds.append(current_preds)
    
    return metric.compute(predictions=decoded_preds, references=decoded_labels)

# [SETTING UP MODEL AND TRAINING ARGUMENTS]
# Create experiment results directory
os.makedirs(EXPERIMENT_RESULTS_DIR_T5_CLS_CE, exist_ok=True)

# Load model
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_T5_CLS_CE)
if checkpoint:
    model = T5ForTokenClassification.from_pretrained(checkpoint)
else:
    model = T5ForTokenClassification.from_pretrained(MODEL_T5_CLS_CE,
                                                     num_labels=NUM_LABELS,
                                                     ignore_mismatched_sizes=True)

# Create training arguments
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_T5_CLS_CE,
    report_to="wandb",
    eval_strategy='steps',
    save_strategy='steps',
    eval_steps=EVAL_STEPS_T5_CLS_CE,
    save_steps=SAVE_STEPS_T5_CLS_CE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_T5_CLS_CE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_T5_CLS_CE,
    num_train_epochs=NUM_TRAIN_EPOCHS_T5_CLS_CE,
    weight_decay=WEIGHT_DECAY_T5_CLS_CE,
    learning_rate=LR_T5_CLS_CE, 
    lr_scheduler_type="linear",  # Linear decay
    output_dir=EXPERIMENT_RESULTS_DIR_T5_CLS_CE,
    logging_dir=EXPERIMENT_RESULTS_DIR_T5_CLS_CE + "/logs",
    logging_steps=LOGGING_STEPS_T5_CLS_CE,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_T5_CLS_CE,
    seed=SEED
)

def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)

# Create optimizer
optimizer = AdamW(model.parameters(), lr=LR_T5_CLS_CE)

# Calculate total training steps
total_steps = len(train_dataset) * training_args.num_train_epochs  # Ensure this is accurate

# Custom linear scheduler with min_lr to prevent decay to 0
class LinearDecayWithMinLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, min_lr, max_steps, last_epoch=-1):
        self.min_lr = min_lr
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        # Linear decay formula with min_lr
        lr_decay = max(0, (1 - step / self.max_steps)) * (self.base_lrs[0] - self.min_lr) + self.min_lr
        return [lr_decay] * len(self.base_lrs)

# Create the custom learning rate scheduler
scheduler = LinearDecayWithMinLR(
    optimizer,
    min_lr=1e-6,  # Set the minimum learning rate to avoid decay to 0
    max_steps=total_steps
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)  # Pass optimizer and scheduler to the Trainer
)

# [TRAINING]
if checkpoint:
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

# [EVALUATING]
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

# [SAVING THINGS]
# Save the model and tokenizer
model.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CLS_CE)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CLS_CE)

# Save the training arguments
with open(EXPERIMENT_RESULTS_DIR_T5_CLS_CE + "/training_args.txt", "w") as f:
    f.write(str(training_args))

# Save the test results
with open(EXPERIMENT_RESULTS_DIR_T5_CLS_CE + "/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to Hugging Face
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=EXPERIMENT_RESULTS_DIR_T5_CLS_CE,
    repo_id="auphong2707/nlp-ner",
    repo_type="model",
    private=False
)
